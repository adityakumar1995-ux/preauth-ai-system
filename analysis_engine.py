import os
import json
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from google import genai
from openai import OpenAI

from rag_engine import retrieve_policy_context, format_context_for_prompt


# =============================================================================
# Environment
# =============================================================================

load_dotenv()


# =============================================================================
# Utility functions
# =============================================================================

def get_secret_value(key: str) -> str:
    """
    Reads secrets from local .env first, then Streamlit secrets if available.
    """
    value = os.getenv(key)

    if value:
        return value

    try:
        import streamlit as st
        return st.secrets.get(key, "")
    except Exception:
        return ""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def clean_json_response(text: str) -> Dict[str, Any]:
    """
    Cleans and parses JSON returned by the LLM.
    """
    if not text:
        raise ValueError("Empty model response.")

    cleaned = text.strip()

    cleaned = re.sub(r"```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in model response: {text[:500]}")

    return json.loads(match.group(0))


def validate_result_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures the model output always has the expected app schema.
    """
    risk_score = data.get("risk_score", 50)

    try:
        risk_score = int(risk_score)
    except Exception:
        risk_score = 50

    risk_score = max(0, min(100, risk_score))

    risk_level = normalize_text(data.get("risk_level", "")).lower()

    if risk_level not in ["low", "medium", "high"]:
        if risk_score < 40:
            risk_level = "low"
        elif risk_score < 70:
            risk_level = "medium"
        else:
            risk_level = "high"

    def ensure_list(value):
        if value is None:
            return []

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        if isinstance(value, str):
            if not value.strip():
                return []

            return [value.strip()]

        return [str(value).strip()]

    validated = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "summary": normalize_text(data.get("summary", "")),
        "denial_reasons": ensure_list(data.get("denial_reasons", [])),
        "missing_documentation": ensure_list(data.get("missing_documentation", [])),
        "recommended_fixes": ensure_list(data.get("recommended_fixes", [])),
    }

    if not validated["summary"]:
        validated["summary"] = (
            "The prior authorization packet was reviewed against the uploaded packet text "
            "and available payer-policy context."
        )

    if not validated["denial_reasons"]:
        if risk_level == "low":
            validated["denial_reasons"] = [
                "No major denial risk found based on the uploaded packet and retrieved policy context."
            ]
        else:
            validated["denial_reasons"] = [
                "Potential documentation or medical-necessity risk was identified."
            ]

    if not validated["recommended_fixes"]:
        if risk_level == "low":
            validated["recommended_fixes"] = [
                "Ensure the complete packet and supporting documentation are attached before submission."
            ]
        else:
            validated["recommended_fixes"] = [
                "Review the missing documentation items and attach supporting records before submission."
            ]

    return validated


# =============================================================================
# Field enrichment for better retrieval/prompt quality
# =============================================================================

def extract_between(raw_text: str, start_label: str, end_labels: List[str]) -> str:
    if not raw_text:
        return ""

    start_match = re.search(re.escape(start_label), raw_text, flags=re.IGNORECASE)

    if not start_match:
        return ""

    start_idx = start_match.end()
    end_idx = len(raw_text)

    for end_label in end_labels:
        end_match = re.search(re.escape(end_label), raw_text[start_idx:], flags=re.IGNORECASE)

        if end_match:
            possible_end = start_idx + end_match.start()
            end_idx = min(end_idx, possible_end)

    return raw_text[start_idx:end_idx].strip()



def remove_evaluation_metadata(raw_text: str) -> str:
    """
    Removes synthetic-evaluation labels and answer-key style sections before the
    LLM sees the packet.

    The AI should evaluate the medical record objectively. It should not use
    labels such as "COMPLETE LOW RISK", "INCOMPLETE HIGH RISK", "Expected Risk",
    or "Known Defects for Synthetic Evaluation" as shortcuts.
    """
    if not raw_text:
        return ""

    metadata_line_patterns = [
        r"\bcomplete\s+low\s+risk\b",
        r"\bincomplete\s+medium\s+risk\b",
        r"\bincomplete\s+high\s+risk\b",
        r"\bexpected\s+(dify\s+)?risk\b",
        r"\bexpected\s+label\b",
        r"\bground\s+truth\b",
        r"\banswer\s+key\b",
        r"\bsynthetic\s+evaluation\b",
        r"\bintentional\s+defects?\b",
        r"\bno\s+intentional\s+defects?\b",
        r"\bknown\s+defects?\b",
        r"\bdesigned\s+as\s+(a\s+)?(low|medium|high)\s+risk\b",
        r"\btest\s+record\s+label\b",
        r"\brisk\s+label\s*:\s*(low|medium|high)\b",
    ]

    section_start_patterns = [
        r"^\s*known\s+defects?\s+for\s+synthetic\s+evaluation\s*:?\s*$",
        r"^\s*synthetic\s+evaluation\s+notes?\s*:?\s*$",
        r"^\s*intentional\s+defects?\s*:?\s*$",
        r"^\s*answer\s+key\s*:?\s*$",
        r"^\s*expected\s+output\s*:?\s*$",
        r"^\s*expected\s+risk\s*:?\s*$",
    ]

    # These are normal clinical/admin section headers where evaluation metadata
    # should stop being skipped if it appeared before them.
    normal_section_patterns = [
        r"^\s*patient\s+information\s*:?\s*$",
        r"^\s*provider\s+information\s*:?\s*$",
        r"^\s*servicing\s+facility\s*:?\s*$",
        r"^\s*requested\s+dos\s*:?\s*$",
        r"^\s*clinical\s+information\s+and\s+codes\s*:?\s*$",
        r"^\s*clinical\s+summary.*:?\s*$",
        r"^\s*treatment\s+timeline\s*:?\s*$",
        r"^\s*relevant\s+medical\s+history\s*:?\s*$",
        r"^\s*past\s+medical\s+history\s*:?\s*$",
        r"^\s*current\s+medications\s*:?\s*$",
        r"^\s*drug\s+allergies\s*:?\s*$",
        r"^\s*physical\s+exam\s*:?\s*$",
        r"^\s*review\s+of\s+systems\s*:?\s*$",
        r"^\s*past\s+imaging\s+and\s+labs\s*:?\s*$",
        r"^\s*pcp\s+referral\s*:?\s*$",
        r"^\s*provider\s+attestation\s*:?\s*$",
        r"^\s*requesting\s+provider\s*:?\s*$",
        r"^\s*facility\s+name\s*:?\s*$",
        r"^\s*primary\s+diagnosis\s*:?\s*$",
        r"^\s*cpt/hcpcs\s+code.*:?\s*$",
    ]

    cleaned_lines = []
    skipping_metadata_section = False

    for line in raw_text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in section_start_patterns):
            skipping_metadata_section = True
            continue

        if skipping_metadata_section:
            if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in normal_section_patterns):
                skipping_metadata_section = False
            else:
                continue

        if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in metadata_line_patterns):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned


def enrich_patient_data_from_raw_text(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Improves weak PDF extraction fields by using objective content from the full
    packet text.

    This function does not classify risk and does not use synthetic-evaluation
    labels such as COMPLETE/INCOMPLETE or expected risk labels.
    """
    enriched = dict(patient_data)
    raw_text_original = normalize_text(enriched.get("raw_text", ""))
    raw_text = remove_evaluation_metadata(raw_text_original)

    if raw_text:
        enriched["raw_text"] = raw_text

    if not raw_text:
        return enriched

    current_service = normalize_text(
        enriched.get("drug_name")
        or enriched.get("procedure")
        or enriched.get("requested_service")
    )

    suspicious_service_values = [
        "",
        "and codes",
        "codes",
        "requested service",
        "drug / procedure / requested service",
        "unknown service",
    ]

    is_suspicious_service = (
        current_service.lower() in suspicious_service_values
        or len(current_service) < 6
    )

    # Pull requested service from objective procedure/code lines only.
    # Do not use synthetic test titles or risk labels as service values.
    cpt_patterns = [
        r"CPT/HCPCS Code\(s\) Requested\s+([0-9A-Z]{4,5}\s*[-–]\s*[^\n]+)",
        r"CPT/HCPCS Code\(s\)\s+Requested\s+([0-9A-Z]{4,5}\s*[-–]\s*[^\n]+)",
        r"CPT Code\s+([0-9A-Z]{4,5}\s*[-–]\s*[^\n]+)",
        r"HCPCS Code\s+([0-9A-Z]{4,5}\s*[-–]\s*[^\n]+)",
        r"Requested Procedure\s*:?\s*([^\n]+)",
        r"Requested Service\s*:?\s*([^\n]+)",
        r"Procedure Requested\s*:?\s*([^\n]+)",
    ]

    extracted_service = ""

    for pattern in cpt_patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)

        if match:
            extracted_service = match.group(1).strip()
            break

    if extracted_service and is_suspicious_service:
        enriched["drug_name"] = extracted_service
        enriched["requested_service"] = extracted_service

    if not normalize_text(enriched.get("diagnosis_code")):
        diagnosis_patterns = [
            r"Primary Diagnosis ICD-10\s+([A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?[^\n]*)",
            r"Primary Diagnosis\s*:?\s*([A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?[^\n]*)",
            r"ICD-10\s*:?\s*([A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?[^\n]*)",
        ]

        for pattern in diagnosis_patterns:
            diagnosis_match = re.search(pattern, raw_text, flags=re.IGNORECASE)

            if diagnosis_match:
                enriched["diagnosis_code"] = diagnosis_match.group(1).strip()
                break

    current_summary = normalize_text(enriched.get("clinical_summary"))

    if len(current_summary) < 80:
        summary = extract_between(
            raw_text=raw_text,
            start_label="Clinical Summary and Letter of Medical Necessity Narrative",
            end_labels=[
                "Treatment Timeline",
                "Relevant Medical History",
                "Physical Exam",
                "Review of Systems",
                "Past Imaging and Labs",
                "PCP Referral",
                "Service-Specific Documentation Checklist",
                "Provider Attestation",
            ],
        )

        if summary:
            enriched["clinical_summary"] = summary[:4000]

    return enriched



# =============================================================================
# Prompt construction
# =============================================================================

def build_ai_review_prompt(patient_data: Dict[str, Any], policy_context: str) -> str:
    """
    Builds the AI review prompt.

    The prompt is intentionally objective:
    - It does not use synthetic answer-key language.
    - It does not ask the model to look for self-labeled defects.
    - It evaluates only the medical record content against payer coverage and
      documentation requirements.
    """
    raw_text = remove_evaluation_metadata(normalize_text(patient_data.get("raw_text", "")))

    patient_fields = {
        "patient_name": patient_data.get("patient_name", ""),
        "insurance_id": patient_data.get("insurance_id", ""),
        "provider_npi": patient_data.get("provider_npi", ""),
        "requested_service_or_procedure": (
            patient_data.get("drug_name", "")
            or patient_data.get("procedure", "")
            or patient_data.get("requested_service", "")
        ),
        "diagnosis_code": patient_data.get("diagnosis_code", ""),
        "clinical_summary": patient_data.get("clinical_summary", ""),
    }

    prompt = f"""
You are an expert prior authorization documentation reviewer for healthcare payer submissions.

Your task:
Objectively review the uploaded prior authorization medical record packet against the retrieved payer coverage, eligibility, and medical-record documentation requirements. Return a structured denial-risk assessment.

Core principle:
Evaluate only the objective clinical, administrative, and medical-record content in the packet. Do not use any answer-key, testing, synthetic-evaluation, or self-assessment language in the packet as evidence.

Do NOT use these as evidence:
- Any label such as "complete", "incomplete", "low risk", "medium risk", "high risk", or "expected risk".
- Any section that describes known defects, intentional defects, synthetic evaluation notes, answer keys, expected output, or ground truth.
- Any sentence that simply tells you the packet is complete, incomplete, inaccurate, outdated, or missing something.
- Any checklist statement by itself as proof that an item is present.

Instead, verify whether the actual medical-record content supports the request.

Use only these evidence sources:
1. The uploaded medical record packet content, including clinical history, diagnosis, physical exam, treatment history, medications, imaging/labs when relevant, referral details, provider/facility details, and requested service/code details.
2. Retrieved payer coverage/eligibility policy excerpts.
3. Retrieved medical-record documentation requirements used for reviews.

Review rules:
1. Read the full medical record packet before deciding anything is missing.
2. Infer the requested procedure/service category from the objective request details, CPT/HCPCS code, diagnosis, and clinical narrative.
3. Compare the medical-record evidence against the relevant payer coverage/eligibility criteria and medical-record documentation requirements.
4. Do not apply one specialty's requirements to another specialty.
5. Do not require imaging, labs, biopsy, physical therapy, or any other documentation unless it is clinically relevant to the requested service or required by the retrieved policy/documentation context.
6. Mark documentation as missing only when the required evidence cannot be found in the objective medical-record content.
7. Mark documentation as weak or incomplete when the record mentions an item but lacks dates, duration, objective findings, results, provider attribution, or clear connection to medical necessity.
8. Do not invent facts that are not in the packet.
9. Do not externally verify IDs, NPI, TIN, member ID, eligibility, or network status. Only flag them if they are absent, blank, internally inconsistent, or contradicted within the packet.
10. If retrieved policy context is unrelated to the request, say so briefly in the summary and rely on objective medical-record evidence plus general documentation-review logic.
11. This is decision support only. Do not approve or deny care. Assess documentation risk for prior authorization submission.

Risk scoring guidance:
- 0 to 39 = low risk
  Use when the packet appears complete, internally consistent, and contains the key medical-record evidence required for the requested service.
- 40 to 69 = medium risk
  Use when the packet has partial, ambiguous, weak, outdated, or minor documentation gaps that may require staff review.
- 70 to 100 = high risk
  Use when the packet lacks major coverage/eligibility evidence, has clear contradictions, missing requested service/code, missing medical-necessity support, or major administrative defects.

Examples of objective specialty-aware review:
- Orthopedic surgery: look for diagnosis, objective exam findings, functional limitation, relevant imaging when required, prior conservative treatment, and treatment plan.
- Bariatric surgery: look for BMI/current height/weight, BMI history, qualifying comorbidities, supervised weight-loss attempts when required, nutrition/psychological evaluation when required, surgical consult, and facility/provider details.
- Gynecology/hysterectomy: look for diagnosis, symptom duration/severity, objective exam when relevant, diagnostic workup when required, failed medical management or contraindications, and clear treatment rationale.
- Medication requests: look for diagnosis, prior therapies, dosing, contraindications, treatment response, and medication history.

Patient/request fields extracted by the app:
{json.dumps(patient_fields, indent=2)}

Objective uploaded medical record packet text:
\"\"\"
{raw_text[:26000]}
\"\"\"

Retrieved payer coverage and medical-record documentation context:
\"\"\"
{policy_context[:14000]}
\"\"\"

Return ONLY valid JSON in this exact schema:
{{
  "risk_score": 0,
  "risk_level": "low",
  "summary": "brief business-facing summary",
  "denial_reasons": ["specific reason 1", "specific reason 2"],
  "missing_documentation": ["specific missing item 1", "specific missing item 2"],
  "recommended_fixes": ["specific actionable fix 1", "specific actionable fix 2"]
}}

Output requirements:
- risk_score must be an integer from 0 to 100.
- risk_level must be one of: low, medium, high.
- If no major denial risk is found, say that clearly.
- If no major missing documentation is found, use an empty list for missing_documentation.
- Tie every denial reason and missing-documentation item to objective medical-record evidence or the absence of objective evidence.
- Recommended fixes should be practical and specific.
- Do not include markdown.
- Do not include text outside JSON.
"""

    return prompt



# =============================================================================
# Source formatting
# =============================================================================

def build_sources(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []

    for chunk in retrieved_chunks:
        sources.append({
            "source_file": chunk.get("source_file"),
            "source_folder": chunk.get("source_folder"),
            "chunk_index": chunk.get("chunk_index"),
            "query_label": chunk.get("query_label"),
        })

    return sources


# =============================================================================
# AI reviewers
# =============================================================================

def groq_ai_review(
    patient_data: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    policy_context: str,
) -> Dict[str, Any]:
    api_key = get_secret_value("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Add it to .env locally or Streamlit secrets in the cloud.")

    model_name = get_secret_value("GROQ_MODEL") or "llama-3.3-70b-versatile"

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = build_ai_review_prompt(
        patient_data=patient_data,
        policy_context=policy_context,
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=1800,
        messages=[
            {
                "role": "system",
                "content": "You are a careful prior authorization documentation reviewer. Use only objective medical-record evidence and payer documentation requirements. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    raw_text = response.choices[0].message.content or ""

    parsed = clean_json_response(raw_text)
    validated = validate_result_schema(parsed)

    validated["retrieved_sources"] = build_sources(retrieved_chunks)
    validated["groq_raw_response"] = raw_text
    validated["analysis_mode"] = "groq_ai_review"

    return validated


def gemini_ai_review(
    patient_data: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    policy_context: str,
) -> Dict[str, Any]:
    api_key = get_secret_value("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Add it to .env locally or Streamlit secrets in the cloud.")

    client = genai.Client(api_key=api_key)

    prompt = build_ai_review_prompt(
        patient_data=patient_data,
        policy_context=policy_context,
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    raw_text = response.text or ""

    parsed = clean_json_response(raw_text)
    validated = validate_result_schema(parsed)

    validated["retrieved_sources"] = build_sources(retrieved_chunks)
    validated["gemini_raw_response"] = raw_text
    validated["analysis_mode"] = "gemini_ai_review"

    return validated


# =============================================================================
# Local fallback
# =============================================================================

def local_fallback_review(
    patient_data: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Last-resort fallback when live AI APIs are unavailable.

    This fallback intentionally avoids using synthetic labels, expected-risk text,
    known-defect sections, or any self-acknowledgement that a document is complete
    or incomplete. It only performs basic objective completeness checks.
    """
    raw_text = remove_evaluation_metadata(normalize_text(patient_data.get("raw_text", "")))

    combined_text = f"""
    {patient_data.get("patient_name", "")}
    {patient_data.get("insurance_id", "")}
    {patient_data.get("provider_npi", "")}
    {patient_data.get("drug_name", "")}
    {patient_data.get("diagnosis_code", "")}
    {patient_data.get("clinical_summary", "")}
    {raw_text}
    """.lower()

    missing = []
    reasons = []
    fixes = []

    service = normalize_text(
        patient_data.get("drug_name")
        or patient_data.get("procedure")
        or patient_data.get("requested_service")
    )

    diagnosis = normalize_text(patient_data.get("diagnosis_code"))
    clinical_summary = normalize_text(patient_data.get("clinical_summary"))

    if not service or service.lower() in ["and codes", "codes", "unknown service"]:
        missing.append("Requested service/procedure and CPT/HCPCS code are not clearly identified.")
        reasons.append("The requested service may not be clearly identifiable from the objective request fields.")
        fixes.append("Add the requested procedure/service name and CPT/HCPCS code before submission.")

    if not diagnosis:
        missing.append("Primary diagnosis code is not clearly documented.")
        reasons.append("The diagnosis supporting the request may be missing or unclear.")
        fixes.append("Add the primary ICD-10 diagnosis code and any relevant secondary diagnoses.")

    if len(clinical_summary) < 80 and len(raw_text) < 800:
        missing.append("Clinical summary or medical-necessity narrative is too limited for review.")
        reasons.append("The packet may not provide enough objective clinical detail to support medical necessity.")
        fixes.append("Add a clinical summary describing diagnosis, history, prior treatment, objective findings, functional impact where relevant, and treatment plan.")

    objective_evidence_terms = [
        "physical exam",
        "examination",
        "assessment",
        "plan",
        "treatment",
        "medication",
        "therapy",
        "imaging",
        "lab",
        "diagnosis",
        "history",
        "symptom",
        "referral",
    ]

    evidence_hits = sum(1 for term in objective_evidence_terms if term in combined_text)

    if evidence_hits < 3:
        missing.append("Objective supporting clinical evidence is limited in the extracted packet text.")
        reasons.append("The packet contains limited objective medical-record evidence for prior authorization review.")
        fixes.append("Attach supporting clinical notes, diagnostic evidence when relevant, treatment history, and provider assessment/plan.")

    if len(missing) >= 3:
        risk_score = 80
        risk_level = "high"
    elif len(missing) >= 1:
        risk_score = 55
        risk_level = "medium"
    else:
        risk_score = 30
        risk_level = "low"
        reasons = [
            "No major objective completeness gaps were detected by the local fallback review."
        ]
        fixes = [
            "Use Groq AI Review Mode for a full payer-policy and medical-record documentation review."
        ]

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "summary": (
            "The packet was reviewed using local fallback logic because live AI review was unavailable. "
            "This fallback performs only basic objective completeness checks and should not be treated as the primary model result."
        ),
        "denial_reasons": reasons,
        "missing_documentation": missing,
        "recommended_fixes": fixes,
        "retrieved_sources": build_sources(retrieved_chunks),
        "analysis_mode": "local_fallback",
    }



# =============================================================================
# Main public function used by app.py
# =============================================================================

def analyze_prior_auth(
    patient_data: Dict[str, Any],
    demo_mode: bool = False,
    analysis_mode: str = "Groq AI Review Mode",
) -> Dict[str, Any]:
    """
    Main analysis function.

    Preferred:
    - Groq AI Review Mode

    Backup:
    - Gemini Live Mode

    Last-resort fallback:
    - Local Fallback Mode
    """
    enriched_patient_data = enrich_patient_data_from_raw_text(patient_data)
    if enriched_patient_data.get("raw_text"):
        enriched_patient_data["raw_text"] = remove_evaluation_metadata(enriched_patient_data["raw_text"])

    retrieved_chunks = []

    try:
        retrieved_chunks = retrieve_policy_context(
            patient_data=enriched_patient_data,
            n_results=12,
            final_k=8,
        )
        policy_context = format_context_for_prompt(retrieved_chunks)
    except Exception as retrieval_error:
        policy_context = (
            "Policy retrieval was unavailable or returned an error. "
            f"Retrieval error: {str(retrieval_error)}"
        )
        retrieved_chunks = []

    if analysis_mode is None:
        analysis_mode = "Groq AI Review Mode"

    # Backward compatibility for old app calls
    if analysis_mode not in ["Groq AI Review Mode", "Gemini Live Mode", "Local Fallback Mode", "Local Mode"]:
        analysis_mode = "Local Fallback Mode" if demo_mode else "Groq AI Review Mode"

    if analysis_mode == "Local Mode":
        analysis_mode = "Local Fallback Mode"

    if analysis_mode == "Groq AI Review Mode":
        try:
            result = groq_ai_review(
                patient_data=enriched_patient_data,
                retrieved_chunks=retrieved_chunks,
                policy_context=policy_context,
            )
            result["policy_context_preview"] = policy_context[:2500]
            return result

        except Exception as groq_error:
            # Try Gemini backup before local fallback
            try:
                result = gemini_ai_review(
                    patient_data=enriched_patient_data,
                    retrieved_chunks=retrieved_chunks,
                    policy_context=policy_context,
                )
                result["analysis_mode"] = "gemini_backup_after_groq_error"
                result["groq_error"] = str(groq_error)
                result["policy_context_preview"] = policy_context[:2500]
                return result

            except Exception as gemini_error:
                fallback = local_fallback_review(
                    patient_data=enriched_patient_data,
                    retrieved_chunks=retrieved_chunks,
                )
                fallback["analysis_mode"] = "fallback_local_rules"
                fallback["groq_error"] = str(groq_error)
                fallback["gemini_error"] = str(gemini_error)
                fallback["policy_context_preview"] = policy_context[:2500]
                return fallback

    if analysis_mode == "Gemini Live Mode":
        try:
            result = gemini_ai_review(
                patient_data=enriched_patient_data,
                retrieved_chunks=retrieved_chunks,
                policy_context=policy_context,
            )
            result["policy_context_preview"] = policy_context[:2500]
            return result

        except Exception as gemini_error:
            fallback = local_fallback_review(
                patient_data=enriched_patient_data,
                retrieved_chunks=retrieved_chunks,
            )
            fallback["analysis_mode"] = "fallback_local_rules"
            fallback["gemini_error"] = str(gemini_error)
            fallback["policy_context_preview"] = policy_context[:2500]
            return fallback

    # Local fallback only
    result = local_fallback_review(
        patient_data=enriched_patient_data,
        retrieved_chunks=retrieved_chunks,
    )
    result["policy_context_preview"] = policy_context[:2500]
    return result


def pretty_print_result(result: Dict[str, Any]) -> None:
    print(json.dumps(result, indent=2))

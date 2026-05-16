import json
import os
import re
from typing import Dict, List, Any

from dotenv import load_dotenv
from google import genai

from rag_engine import retrieve_policy_context, format_context_for_prompt


# =============================================================================
# Environment setup
# =============================================================================

load_dotenv()


# =============================================================================
# Helper functions
# =============================================================================

def risk_level_from_score(score: int) -> str:
    if score < 40:
        return "low"
    if score < 70:
        return "medium"
    return "high"


def contains_any(text: str, terms: list[str]) -> bool:
    """
    Safer keyword matching.

    Short terms use word-boundary matching so:
    - PT does not match CPT
    - CT does not match meniscectomy
    """
    text_lower = text.lower()

    for term in terms:
        term_lower = term.lower().strip()

        if not term_lower:
            continue

        if len(term_lower) <= 3:
            pattern = r"\b" + re.escape(term_lower) + r"\b"
            if re.search(pattern, text_lower):
                return True
        else:
            if term_lower in text_lower:
                return True

    return False


def clean_json_response(text: str) -> Dict[str, Any]:
    """
    Cleans and parses JSON returned by Gemini.
    """
    if not text:
        raise ValueError("Empty Gemini response")

    cleaned = text.strip()
    cleaned = re.sub(r"```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if not match:
        raise ValueError("No valid JSON object found in Gemini response")

    return json.loads(match.group(0))


def validate_result_schema(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures Gemini output has the keys our Streamlit app expects.
    """
    risk_score = result.get("risk_score", 50)

    try:
        risk_score = int(risk_score)
    except Exception:
        risk_score = 50

    risk_score = max(0, min(100, risk_score))

    risk_level = str(
        result.get("risk_level", risk_level_from_score(risk_score))
    ).lower()

    if risk_level not in ["low", "medium", "high"]:
        risk_level = risk_level_from_score(risk_score)

    denial_reasons = result.get("denial_reasons", [])
    missing_documentation = result.get("missing_documentation", [])
    recommended_fixes = result.get("recommended_fixes", [])
    policy_references = result.get("policy_references", [])

    if not isinstance(denial_reasons, list):
        denial_reasons = []

    if not isinstance(missing_documentation, list):
        missing_documentation = []

    if not isinstance(recommended_fixes, list):
        recommended_fixes = []

    if not isinstance(policy_references, list):
        policy_references = []

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "summary": str(result.get("summary", "")),
        "denial_reasons": denial_reasons,
        "missing_documentation": missing_documentation,
        "recommended_fixes": recommended_fixes,
        "policy_references": policy_references,
    }


# =============================================================================
# Critical gap detection
# =============================================================================

def detect_critical_gaps(patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detects explicit prior authorization gaps from the full raw packet text.

    These are not tied to specific synthetic record numbers.
    They are general denial-risk patterns.
    """

    procedure = (
        patient_data.get("drug_name", "")
        or patient_data.get("procedure", "")
        or patient_data.get("requested_service", "")
    )

    diagnosis = patient_data.get("diagnosis_code", "")
    clinical_summary = patient_data.get("clinical_summary", "")
    raw_text = patient_data.get("raw_text", "")

    combined_text = f"""
    {procedure}
    {diagnosis}
    {clinical_summary}
    {raw_text}
    """.lower()

    gaps = []

    def add_gap(code, severity, label, reason, fix, weight):
        if not any(gap["code"] == code for gap in gaps):
            gaps.append(
                {
                    "code": code,
                    "severity": severity,  # blocker, major, medium
                    "label": label,
                    "reason": reason,
                    "fix": fix,
                    "weight": weight,
                }
            )

    # -------------------------------------------------------------------------
    # Missing CPT / HCPCS
    # -------------------------------------------------------------------------

    cpt_present = bool(re.search(r"\b\d{5}\b", procedure)) or bool(
        re.search(r"\bcpt\s*(code)?\s*[:\-]?\s*\d{5}\b", combined_text)
    )

    cpt_missing_signals = [
        "cpt/hcpcs code(s) requested not provided",
        "cpt code missing",
        "requested cpt/hcpcs code no",
        "no cpt code",
        "cpt code and pcp referral missing",
        "cpt/hcpcs code(s) requested no",
    ]

    if not cpt_present and any(signal in combined_text for signal in cpt_missing_signals):
        add_gap(
            code="missing_cpt",
            severity="blocker",
            label="Requested CPT/HCPCS code missing",
            reason="The packet does not provide a requested CPT/HCPCS code, which is a critical authorization defect.",
            fix="Add the exact requested CPT/HCPCS code for the procedure before submission.",
            weight=25,
        )

    # -------------------------------------------------------------------------
    # Missing signed LMN
    # -------------------------------------------------------------------------

    lmn_missing_signals = [
        "letter of medical necessity narrative: not included",
        "signed lmn no",
        "signed lmn missing",
        "no separate signed lmn",
        "lmn no",
        "letter of medical necessity not included",
        "not included as a separate signed letter",
        "surgical rationale appears only briefly",
    ]

    if any(signal in combined_text for signal in lmn_missing_signals):
        add_gap(
            code="missing_lmn",
            severity="blocker",
            label="Signed letter of medical necessity missing",
            reason="The packet indicates that a signed letter of medical necessity is not included.",
            fix="Attach a signed letter of medical necessity explaining the clinical rationale and payer-policy support.",
            weight=20,
        )

    # -------------------------------------------------------------------------
    # Missing PCP referral
    # -------------------------------------------------------------------------

    pcp_missing_signals = [
        "referring pcp not included",
        "pcp npi not included",
        "referral date not included",
        "referral reason not included",
        "pcp referral no",
        "pcp referral absent",
        "pcp referral is absent",
        "pcp referral missing",
        "referral reason not included in packet",
    ]

    if any(signal in combined_text for signal in pcp_missing_signals):
        add_gap(
            code="missing_pcp_referral",
            severity="major",
            label="PCP referral missing",
            reason="The packet indicates that PCP referral information is absent or incomplete.",
            fix="Attach PCP referral details including referring PCP, NPI, referral date, and referral reason.",
            weight=15,
        )

    # -------------------------------------------------------------------------
    # Missing DOS / start / end date
    # -------------------------------------------------------------------------

    dos_missing_signals = [
        "requested dos: not provided",
        "start date not provided",
        "end date not provided",
        "requested dos/start/end no",
        "lacks requested date of service",
        "requested service dates",
    ]

    if any(signal in combined_text for signal in dos_missing_signals):
        add_gap(
            code="missing_dos",
            severity="blocker",
            label="Requested DOS/start/end date missing",
            reason="The packet does not provide complete requested date-of-service/start/end date information.",
            fix="Add requested DOS, start date, and end date before submission.",
            weight=20,
        )

    # -------------------------------------------------------------------------
    # Missing / unsigned attestation
    # -------------------------------------------------------------------------

    attestation_missing_signals = [
        "provider attestation/signature no",
        "signature/date missing",
        "provider signature not signed",
        "date of signature not dated",
        "attestation section present but not completed",
    ]

    if any(signal in combined_text for signal in attestation_missing_signals):
        add_gap(
            code="missing_attestation",
            severity="blocker",
            label="Provider attestation/signature missing",
            reason="The packet indicates the provider attestation or signature/date is missing or incomplete.",
            fix="Complete the provider attestation with provider signature and date.",
            weight=20,
        )

    # -------------------------------------------------------------------------
    # Incomplete / outdated imaging
    # -------------------------------------------------------------------------

    imaging_problem_signals = [
        "current imaging no",
        "mri is from 09/2023",
        "outdated mri",
        "report is over 2.5 years old",
        "no imaging report attached",
        "mri referenced but report/impression not attached",
        "actual report/impression is not included",
        "referenced in note but actual report/impression not included",
        "full report not attached",
        "report/impression missing",
        "imaging report no",
        "no weight-bearing x-ray report",
        "angles/measurements not provided",
        "no report in packet",
        "mr arthrogram is referenced but actual report is incomplete",
    ]

    if any(signal in combined_text for signal in imaging_problem_signals):
        add_gap(
            code="imaging_problem",
            severity="major",
            label="Imaging report missing, incomplete, or outdated",
            reason="The packet indicates imaging is missing, incomplete, outdated, or lacks required measurements/impression.",
            fix="Attach current diagnostic imaging report with interpretation, impression, and relevant measurements/findings.",
            weight=20,
        )

    # -------------------------------------------------------------------------
    # Conservative treatment missing/incomplete
    # -------------------------------------------------------------------------

    conservative_problem_signals = [
        "prior therapies tried/failed with dates no",
        "no recent pt/injection timeline included",
        "no recent pt notes",
        "conservative treatment details not documented",
        "prior therapies partial",
        "no formal pt/bracing timeline",
        "no formal pt/bracing timeline after reinjury",
        "footwear/orthotic trial documentation",
        "no footwear/orthotic/pt documentation",
        "corticosteroid injection trial/contraindication no",
        "no injection note or contraindication in packet",
    ]

    if any(signal in combined_text for signal in conservative_problem_signals):
        add_gap(
            code="incomplete_conservative_treatment",
            severity="medium",
            label="Conservative treatment documentation missing or incomplete",
            reason="The packet indicates prior conservative therapy documentation is missing, partial, or lacks dates/duration/outcomes.",
            fix="Add a conservative-treatment timeline with dates, duration, therapies tried, outcomes, and reason for failure or contraindication.",
            weight=10,
        )

    # -------------------------------------------------------------------------
    # Functional score / ADL impact missing or partial
    # -------------------------------------------------------------------------

    functional_problem_signals = [
        "ikdc score not provided",
        "koos not provided",
        "functional disability scale no",
        "functional score not provided",
        "no ases/quickdash functional score",
        "pain severity/functional impairment partial",
        "functional impact vague",
        "functional impact is described generally but not fully quantified",
        "pain and adl impact partial",
        "impact not detailed",
        "current functional disability scale",
        "no complete rom/functional score",
    ]

    if any(signal in combined_text for signal in functional_problem_signals):
        add_gap(
            code="functional_evidence_incomplete",
            severity="medium",
            label="Functional limitation evidence missing or incomplete",
            reason="The packet indicates functional impact or disability scoring is missing, vague, or incomplete.",
            fix="Add specific ADL/work limitations and, when applicable, a functional score such as KOOS, IKDC, Lysholm, ASES, QuickDASH, HOOS, or WOMAC.",
            weight=10,
        )

    # -------------------------------------------------------------------------
    # Treatment plan / rehab ability incomplete
    # -------------------------------------------------------------------------

    treatment_plan_problem_signals = [
        "treatment plan partial",
        "surgery discussed, rehab plan not documented",
        "ability to rehab no",
        "treatment plan/pre-op discussion partial",
        "discussion limited",
        "surgical rationale appears only briefly",
    ]

    if any(signal in combined_text for signal in treatment_plan_problem_signals):
        add_gap(
            code="treatment_plan_incomplete",
            severity="medium",
            label="Treatment plan or rehabilitation rationale incomplete",
            reason="The packet indicates the treatment plan, surgical rationale, or rehabilitation plan is incomplete.",
            fix="Add clear surgical rationale, pre-op discussion, expected recovery, and ability to participate in rehabilitation.",
            weight=10,
        )

    # -------------------------------------------------------------------------
    # Laterality / coding mismatch
    # -------------------------------------------------------------------------

    laterality_problem_signals = [
        "procedure right wrist; primary icd-10 states left wrist",
        "laterality mismatch",
        "laterality consistency no",
        "primary diagnosis code specifies left wrist",
        "procedure right, primary code left",
        "laterality inconsistency remains unresolved",
    ]

    if any(signal in combined_text for signal in laterality_problem_signals):
        add_gap(
            code="laterality_mismatch",
            severity="blocker",
            label="Laterality mismatch between procedure and diagnosis",
            reason="The packet contains a laterality mismatch between the requested procedure and diagnosis code.",
            fix="Correct the diagnosis/procedure laterality so the requested service, ICD-10 code, imaging, and exam all match.",
            weight=25,
        )

    # -------------------------------------------------------------------------
    # Demographic mismatch
    # -------------------------------------------------------------------------

    demographic_problem_signals = [
        "demographic accuracy no",
        "dob and age mismatch",
        "patient dob suggests age",
        "packet lists age",
    ]

    if any(signal in combined_text for signal in demographic_problem_signals):
        add_gap(
            code="demographic_mismatch",
            severity="major",
            label="Patient demographic inconsistency",
            reason="The packet contains inconsistent demographic information such as DOB/age mismatch.",
            fix="Correct demographic fields before submission to avoid administrative denial or rework.",
            weight=15,
        )

    # -------------------------------------------------------------------------
    # Revision surgery details
    # -------------------------------------------------------------------------

    is_revision_case = contains_any(
        combined_text,
        [
            "revision",
            "prior acl reconstruction",
            "acl graft",
            "graft failure",
            "left knee revision acl reconstruction",
        ],
    )

    revision_missing_signals = [
        "complete staged revision plan",
        "staged revision plan",
        "graft choice not documented",
        "revision plan details partial",
        "complete staged revision plan missing",
        "staged plan and graft choice not documented",
        "does not provide cpt code, prior acl surgery date, complete staged revision plan",
    ]

    if is_revision_case and any(signal in combined_text for signal in revision_missing_signals):
        add_gap(
            code="incomplete_revision_plan",
            severity="blocker",
            label="Incomplete revision surgery plan",
            reason="The packet suggests a revision ACL procedure but does not include a complete staged revision plan or graft-choice details.",
            fix="Add a complete revision surgery plan, including staged approach if applicable, graft choice, surgical rationale, and expected rehabilitation plan.",
            weight=20,
        )

    prior_surgery_missing_signals = [
        "prior acl reconstruction date and operative report not included",
        "prior acl surgery date",
        "operative report not included",
        "date and operative report not included",
        "prior surgery date/history partial",
        "date, graft type, and operative report not included",
        "prior acl noted, date/operative report missing",
        "prior acl noted, date and operative report missing",
        "prior acl reconstruction - date, graft type, and operative report not included",
    ]

    if is_revision_case and any(signal in combined_text for signal in prior_surgery_missing_signals):
        add_gap(
            code="missing_prior_surgery_details",
            severity="major",
            label="Prior surgery details missing",
            reason="The packet references prior ACL reconstruction but does not include the prior surgery date, graft type, or operative report.",
            fix="Attach prior ACL operative report, surgery date, graft type, and relevant prior reconstruction details.",
            weight=15,
        )

    return gaps


# =============================================================================
# Score calibration
# =============================================================================

def apply_critical_gaps_to_result(
    result: Dict[str, Any],
    critical_gaps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Adds detected gaps and recalibrates score.

    This avoids forcing one medium issue into high risk.
    It also prevents complete packets from receiving unrealistic 0-risk scores.
    """

    current_score = int(result.get("risk_score", 50))

    # -------------------------------------------------------------------------
    # If no critical gaps exist, keep Gemini's judgment but prevent unrealistic 0.
    # -------------------------------------------------------------------------

    if not critical_gaps:
        result["critical_gaps"] = []

        if current_score < 10:
            current_score = 10

        current_score = min(current_score, 35) if result.get("risk_level") == "low" else current_score

        result["risk_score"] = current_score
        result["risk_level"] = risk_level_from_score(current_score)

        return result

    blocker_count = sum(1 for gap in critical_gaps if gap.get("severity") == "blocker")
    major_count = sum(1 for gap in critical_gaps if gap.get("severity") == "major")
    medium_count = sum(1 for gap in critical_gaps if gap.get("severity") == "medium")

    adjusted_score = current_score

    # Add softer weights.
    for gap in critical_gaps:
        severity = gap.get("severity")

        if severity == "blocker":
            adjusted_score += min(int(gap.get("weight", 20)), 20)
        elif severity == "major":
            adjusted_score += min(int(gap.get("weight", 15)), 15)
        else:
            adjusted_score += min(int(gap.get("weight", 10)), 8)

    adjusted_score = min(adjusted_score, 95)

    # -------------------------------------------------------------------------
    # Risk floors based on gap severity.
    # -------------------------------------------------------------------------

    if blocker_count >= 3:
        adjusted_score = max(adjusted_score, 90)
    elif blocker_count == 2:
        adjusted_score = max(adjusted_score, 80)
    elif blocker_count == 1 and major_count >= 1:
        adjusted_score = max(adjusted_score, 75)
    elif blocker_count == 1:
        adjusted_score = max(adjusted_score, 65)

    elif major_count >= 3:
        adjusted_score = max(adjusted_score, 80)
    elif major_count == 2:
        adjusted_score = max(adjusted_score, 70)
    elif major_count == 1 and medium_count >= 2:
        adjusted_score = max(adjusted_score, 65)
    elif major_count == 1:
        adjusted_score = max(adjusted_score, 55)

    elif medium_count >= 3:
        adjusted_score = max(adjusted_score, 65)
    elif medium_count == 2:
        adjusted_score = max(adjusted_score, 55)
    elif medium_count == 1:
        adjusted_score = max(adjusted_score, 40)

    result["risk_score"] = adjusted_score
    result["risk_level"] = risk_level_from_score(adjusted_score)
    result["critical_gaps"] = critical_gaps

    for gap in critical_gaps:
        reason = gap["reason"]
        label = gap["label"]
        fix = gap["fix"]

        if reason not in result.get("denial_reasons", []):
            result.setdefault("denial_reasons", []).append(reason)

        if label not in result.get("missing_documentation", []):
            result.setdefault("missing_documentation", []).append(label)

        if fix not in result.get("recommended_fixes", []):
            result.setdefault("recommended_fixes", []).append(fix)

    return result


# =============================================================================
# Demo / rule-based analysis engine
# =============================================================================

def demo_analysis(
    patient_data: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Demo-mode prior authorization analysis.
    Uses clinical summary + full raw packet text.
    """

    procedure = (
        patient_data.get("drug_name", "")
        or patient_data.get("procedure", "")
        or patient_data.get("requested_service", "")
    )

    diagnosis = patient_data.get("diagnosis_code", "")
    clinical_summary = patient_data.get("clinical_summary", "")
    raw_text = patient_data.get("raw_text", "")

    combined_text = f"""
    {procedure}
    {diagnosis}
    {clinical_summary}
    {raw_text}
    """.lower()

    missing_documentation = []
    denial_reasons = []
    recommended_fixes = []

    risk_score = 30

    imaging_terms = [
        "mri",
        "x-ray",
        "xray",
        "x ray",
        "ct scan",
        "ultrasound",
        "imaging report",
        "diagnostic image",
        "diagnostic imaging",
        "radiology report",
        "radiograph",
        "mri arthrogram",
        "mr arthrogram",
        "weight-bearing x-ray",
    ]

    if not contains_any(combined_text, imaging_terms):
        risk_score += 20
        missing_documentation.append(
            "Recent diagnostic imaging report, such as MRI, X-ray, CT scan, ultrasound, or other relevant imaging."
        )
        denial_reasons.append(
            "The request may be denied if diagnostic imaging evidence supporting the requested service is not clearly documented."
        )
        recommended_fixes.append(
            "Attach the most recent imaging report and clearly reference the abnormal findings that support the requested procedure."
        )

    conservative_terms = [
        "physical therapy",
        "supervised pt",
        "pt visits",
        "nsaid",
        "nsaids",
        "anti-inflammatory",
        "anti inflammatory",
        "medication",
        "medications",
        "activity modification",
        "home exercise",
        "home exercise program",
        "injection",
        "injections",
        "corticosteroid",
        "steroid injection",
        "bracing",
        "brace",
        "rest",
        "ice",
        "heat",
        "conservative treatment",
        "conservative therapy",
        "failed conservative",
        "failed therapy",
        "tried and failed",
        "eight weeks",
        "8 weeks",
        "six weeks",
        "6 weeks",
        "10 weeks",
        "12 weeks",
        "14 visits",
        "18 visits",
        "shoe modification",
        "orthotic",
        "splinting",
        "throwing cessation",
    ]

    if not contains_any(combined_text, conservative_terms):
        risk_score += 20
        missing_documentation.append(
            "Documentation of conservative treatment tried, failed, or contraindicated, including treatment type, dates, duration, response, and reason for discontinuation."
        )
        denial_reasons.append(
            "The payer may consider the request insufficient if conservative treatment history is missing or not specific enough."
        )
        recommended_fixes.append(
            "Add a clear conservative-treatment timeline, such as physical therapy dates, medications tried, activity modification, home exercise program, injections, bracing, and response to each."
        )

    function_terms = [
        "functional limitation",
        "functional limitations",
        "functional impairment",
        "functional disability",
        "adl",
        "adls",
        "activities of daily living",
        "difficulty walking",
        "limited mobility",
        "unable to walk",
        "stairs",
        "difficulty climbing stairs",
        "kneeling",
        "locking",
        "catching",
        "reduced walking tolerance",
        "limping",
        "antalgic gait",
        "work limitation",
        "night pain",
        "weakness affecting work duties",
        "overhead limitations",
        "reduced overhead reach",
        "unable to teach fitness classes",
        "difficulty squatting",
        "prolonged sitting",
        "cannot coach running drills",
        "functional impact",
        "koos-adl",
        "lysholm",
        "ikdc",
        "ases",
        "quickdash",
        "hoos-adl",
        "womac",
    ]

    if not contains_any(combined_text, function_terms):
        risk_score += 15
        missing_documentation.append(
            "Severity of pain and specific functional limitations affecting activities of daily living."
        )
        denial_reasons.append(
            "The request may lack evidence that symptoms significantly affect daily function beyond general pain."
        )
        recommended_fixes.append(
            "Document specific functional impact, such as difficulty walking, stairs, work limitations, sleep disruption, limping, or reduced activities of daily living."
        )

    physical_exam_terms = [
        "physical exam",
        "physical examination",
        "exam findings",
        "range of motion",
        "rom",
        "tenderness",
        "swelling",
        "effusion",
        "positive test",
        "mcmurray",
        "mcmurray's",
        "lachman",
        "pivot shift",
        "joint line tenderness",
        "instability",
        "drawer",
        "varus",
        "valgus",
        "painful arc",
        "jobe",
        "empty can",
        "neer",
        "hawkins",
        "fadir",
        "impingement",
        "anterior drawer",
        "talar tilt",
        "ulnar fovea tenderness",
        "grip strength",
    ]

    if not contains_any(combined_text, physical_exam_terms):
        risk_score += 10
        missing_documentation.append(
            "Pertinent physical examination findings related to the requested service."
        )
        denial_reasons.append(
            "The payer may find the request unsupported if relevant physical examination findings are not included."
        )
        recommended_fixes.append(
            "Add relevant physical exam findings, such as range of motion, tenderness, swelling, instability, special tests, or other procedure-specific findings."
        )

    if not diagnosis.strip():
        risk_score += 10
        missing_documentation.append("ICD-10 diagnosis code supporting the requested service.")
        denial_reasons.append("The request may be denied if diagnosis information is incomplete or missing.")
        recommended_fixes.append("Add all applicable ICD-10 diagnosis codes that support the requested service.")

    procedure_terms = [
        "cpt",
        "hcpcs",
        "procedure",
        "surgery",
        "arthroscopy",
        "injection",
        "implant",
        "therapy",
        "scan",
        "meniscectomy",
        "repair",
        "reconstruction",
        "bunionectomy",
        "osteotomy",
        "debridement",
    ]

    has_clear_service = bool(procedure.strip()) and contains_any(
        procedure.lower(),
        procedure_terms
    )

    if not has_clear_service:
        risk_score += 10
        missing_documentation.append("Requested procedure, CPT/HCPCS code, or clear service description.")
        denial_reasons.append("The requested service may not be clearly identified.")
        recommended_fixes.append("Add the requested procedure name and CPT/HCPCS code where applicable.")

    treatment_plan_terms = [
        "treatment plan",
        "pre-op",
        "preoperative",
        "surgical plan",
        "recommended surgery",
        "provider recommends",
        "physician recommends",
        "medical necessity",
        "letter of medical necessity",
        "rationale",
        "plan is",
        "scheduled for",
        "surgical planning",
        "arthroscopy planned",
        "arthroscopic repair recommended",
        "arthroscopic approach recommended",
        "ligament reconstruction discussed",
        "post-operative rehabilitation",
        "rehab plan",
    ]

    if not contains_any(combined_text, treatment_plan_terms):
        risk_score += 5
        missing_documentation.append(
            "Physician treatment plan or rationale for the requested service."
        )
        denial_reasons.append(
            "The submission may be weaker if it does not clearly explain the provider's treatment plan and rationale."
        )
        recommended_fixes.append(
            "Include the provider's treatment plan, rationale for the requested service, and why it is medically necessary at this time."
        )

    risk_score = min(risk_score, 95)
    risk_level = risk_level_from_score(risk_score)

    if not missing_documentation:
        risk_score = 25
        risk_level = "low"
        denial_reasons.append("No major denial risk found based on the available request summary.")
        recommended_fixes.append("Ensure all supporting clinical records are attached before submission.")

    sources = []

    for chunk in retrieved_chunks:
        sources.append({
            "source_file": chunk.get("source_file"),
            "source_folder": chunk.get("source_folder"),
            "chunk_index": chunk.get("chunk_index"),
            "query_label": chunk.get("query_label"),
        })

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "summary": (
            f"The request for {procedure or 'the requested service'} was reviewed against the retrieved "
            f"payer policy and documentation guidance. The main denial risk depends on whether the submission "
            f"clearly supports medical necessity with diagnostic evidence, conservative treatment history, "
            f"physical examination findings, functional impact, diagnosis support, and a clear treatment plan."
        ),
        "denial_reasons": denial_reasons,
        "missing_documentation": missing_documentation,
        "recommended_fixes": recommended_fixes,
        "retrieved_sources": sources,
    }


# =============================================================================
# Gemini live analysis
# =============================================================================

def build_gemini_prompt(patient_data: Dict[str, Any], policy_context: str) -> str:
    return f"""
You are a prior authorization documentation review assistant for a healthcare payer-policy workflow.

Your task:
Review the prior authorization request against the retrieved payer policy excerpts and documentation guidance.

Important:
- Use only the patient/request details and retrieved policy excerpts provided below.
- Use the raw extracted packet text as the source of truth for what is present, missing, partial, outdated, or inconsistent.
- Do not invent facts.
- If a document is missing, incomplete, outdated, inconsistent, or unclear, mark it as a risk.
- Be practical and payer-review focused.
- Return ONLY valid JSON. No markdown. No explanation outside JSON.

Patient/request details:
Patient name: {patient_data.get("patient_name", "")}
Insurance member ID: {patient_data.get("insurance_id", "")}
Provider NPI: {patient_data.get("provider_npi", "")}
Requested service/procedure/drug: {patient_data.get("drug_name", "")}
Diagnosis code(s): {patient_data.get("diagnosis_code", "")}

Clinical summary:
{patient_data.get("clinical_summary", "")}

Raw extracted packet text:
{patient_data.get("raw_text", "")[:10000]}

Retrieved payer policy and documentation excerpts:
{policy_context}

Return JSON in exactly this structure:
{{
  "risk_score": 0,
  "risk_level": "low | medium | high",
  "summary": "Brief payer-review summary of the request.",
  "denial_reasons": [
    "Specific reason why this request may be denied, tied to missing or weak documentation."
  ],
  "missing_documentation": [
    "Specific missing document or missing clinical detail."
  ],
  "recommended_fixes": [
    "Specific fix the provider/coordinator should make before submission."
  ],
  "policy_references": [
    {{
      "source": "Name of relevant source file if available from context",
      "reason": "Short explanation of why this source matters"
    }}
  ]
}}

Risk scoring guidance:
- 0-9 = exceptional/near-perfect low-complexity packet. Use rarely.
- 10-35 = low risk: key medical necessity evidence appears present.
- 36-69 = medium risk: some important documentation gaps, ambiguity, or partial evidence.
- 70-100 = high risk: major evidence gaps, administrative blockers, inconsistencies, or likely denial triggers.

Important calibration:
- Do not give 0 unless the packet is exceptionally complete and low-complexity.
- A complete prior authorization packet usually still has some baseline review risk.
- One medium documentation weakness should usually be medium risk, not automatically high risk.
- Multiple administrative blockers should be high risk.

Critical high-risk examples:
- Missing CPT/HCPCS code
- Missing requested DOS/start/end date
- Missing provider signature or attestation
- Missing signed letter of medical necessity
- Missing PCP referral when expected in packet
- Missing or outdated imaging
- Laterality mismatch between procedure and diagnosis
- Patient demographic inconsistency
- Missing prior operative report for revision surgery
- Missing staged revision plan or graft choice
- Missing functional score when the packet itself says no score was provided
- Incomplete conservative treatment timeline

When scoring, consider whether the request includes:
- clear requested service and CPT/HCPCS code where applicable
- diagnosis support
- medical necessity rationale
- relevant imaging or diagnostic testing where applicable
- prior conservative treatment tried/failed/contraindicated where applicable
- dates, duration, and outcomes of prior treatment
- physical examination findings
- severity of symptoms and functional limitations/ADL impact
- provider treatment plan or rationale
- payer-specific documentation requirements from the retrieved excerpts
"""


def gemini_analysis(
    patient_data: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    policy_context: str
) -> Dict[str, Any]:

    api_key = os.getenv("GEMINI_API_KEY")

    try:
        import streamlit as st
        api_key = api_key or st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Add it to your .env file or Streamlit secrets.")

    client = genai.Client(api_key=api_key)

    prompt = build_gemini_prompt(
        patient_data=patient_data,
        policy_context=policy_context,
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    raw_text = response.text

    parsed = clean_json_response(raw_text)
    validated = validate_result_schema(parsed)

    sources = []

    for chunk in retrieved_chunks:
        sources.append({
            "source_file": chunk.get("source_file"),
            "source_folder": chunk.get("source_folder"),
            "chunk_index": chunk.get("chunk_index"),
            "query_label": chunk.get("query_label"),
        })

    validated["retrieved_sources"] = sources
    validated["gemini_raw_response"] = raw_text

    return validated


# =============================================================================
# Main analysis function
# =============================================================================

def analyze_prior_auth(
    patient_data: Dict[str, Any],
    demo_mode: bool = True
) -> Dict[str, Any]:

    retrieved_chunks = retrieve_policy_context(
        patient_data=patient_data,
        n_results=10,
        final_k=6,
    )

    policy_context = format_context_for_prompt(retrieved_chunks)
    critical_gaps = detect_critical_gaps(patient_data)

    if demo_mode:
        result = demo_analysis(
            patient_data=patient_data,
            retrieved_chunks=retrieved_chunks,
        )

        result = apply_critical_gaps_to_result(result, critical_gaps)

        result["policy_context_preview"] = policy_context[:2500]
        result["analysis_mode"] = "demo_rules"

        return result

    try:
        result = gemini_analysis(
            patient_data=patient_data,
            retrieved_chunks=retrieved_chunks,
            policy_context=policy_context,
        )

        result = apply_critical_gaps_to_result(result, critical_gaps)

        result["policy_context_preview"] = policy_context[:2500]
        result["analysis_mode"] = "gemini_live"

        return result

    except Exception as e:
        fallback = demo_analysis(
            patient_data=patient_data,
            retrieved_chunks=retrieved_chunks,
        )

        fallback = apply_critical_gaps_to_result(fallback, critical_gaps)

        fallback["policy_context_preview"] = policy_context[:2500]
        fallback["analysis_mode"] = "fallback_demo_rules"
        fallback["gemini_error"] = str(e)

        return fallback


def pretty_print_result(result: Dict[str, Any]) -> None:
    print(json.dumps(result, indent=2))
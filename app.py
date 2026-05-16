import json
import re
import time
from pathlib import Path
from datetime import datetime

import streamlit as st

from analysis_engine import analyze_prior_auth
from pdf_extractor import extract_fields_from_pdf
from report_generator import generate_report_pdf_bytes


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(
    page_title="PreAuth.ai",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RECORDS_FILE = DATA_DIR / "preauth_records.json"


# =============================================================================
# Styling
# =============================================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f5f0;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3350;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .card {
        background: white;
        border: 1px solid #e2ddd6;
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2ddd6;
        padding: 1rem;
        border-radius: 10px;
    }

    .risk-low {
        color: #1a7f37;
        font-weight: 700;
    }

    .risk-medium {
        color: #b26a00;
        font-weight: 700;
    }

    .risk-high {
        color: #c0392b;
        font-weight: 700;
    }

    .source-box {
        background: #f6f6f6;
        border-left: 4px solid #1a3350;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }

    .small-note {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.25rem;
    }

    .success-box {
        background: #e9f7ef;
        border-left: 4px solid #1a7f37;
        padding: 0.9rem;
        margin-bottom: 1rem;
        border-radius: 6px;
    }

    .warning-box {
        background: #fff7e6;
        border-left: 4px solid #b26a00;
        padding: 0.9rem;
        margin-bottom: 1rem;
        border-radius: 6px;
    }

    .danger-box {
        background: #fdecea;
        border-left: 4px solid #c0392b;
        padding: 0.9rem;
        margin-bottom: 1rem;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Data functions
# =============================================================================

def load_records():
    if RECORDS_FILE.exists():
        with open(RECORDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_records(records):
    with open(RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def add_submission(patient_data, result):
    records = load_records()

    member_id = patient_data.get("insurance_id", "UNKNOWN").strip() or "UNKNOWN"
    patient_name = patient_data.get("patient_name", "Unknown Patient").strip() or "Unknown Patient"

    procedure = (
        patient_data.get("drug_name", "")
        or patient_data.get("procedure", "")
        or patient_data.get("requested_service", "")
        or "Unknown Service"
    )

    if member_id not in records:
        records[member_id] = {
            "patient_name": patient_name,
            "member_id": member_id,
            "submissions": [],
        }

    records[member_id]["patient_name"] = patient_name

    records[member_id]["submissions"].append(
        {
            "id": f"PA-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%b %d, %Y"),
            "procedure": procedure,
            "risk_score": result.get("risk_score", 0),
            "risk_level": result.get("risk_level", "unknown"),
            "result": result,
        }
    )

    save_records(records)


# =============================================================================
# UI helper functions
# =============================================================================

def risk_class(level):
    level = str(level).lower()

    if level == "low":
        return "risk-low"

    if level == "medium":
        return "risk-medium"

    return "risk-high"


def clean_filename(value):
    value = value or "patient"
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def render_result(result):
    score = result.get("risk_score", 0)
    level = result.get("risk_level", "unknown")
    css_class = risk_class(level)

    analysis_mode = result.get("analysis_mode", "")

    if analysis_mode == "fallback_local_rules":
        if result.get("gemini_error"):
            st.warning(
                "Gemini Live Mode was unavailable, so the app used Local Mode fallback. "
                "This may happen because of API quota, missing key, or connection issues."
            )

        if result.get("claude_error"):
            st.warning(
                "Claude Live Mode was unavailable, so the app used Local Mode fallback. "
                "This may happen because of API credits, missing key, or connection issues."
            )

    st.markdown(
        f"""
        <div class="card">
            <h2>Analysis Result</h2>
            <h1 class="{css_class}">{score} / 100 — {level.upper()} RISK</h1>
            <p>{result.get("summary", "")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Denial Reasons")
        reasons = result.get("denial_reasons", [])

        if reasons:
            for item in reasons:
                st.warning(item)
        else:
            st.success("No major denial reasons found.")

    with col2:
        st.markdown("### Missing Documentation")
        missing = result.get("missing_documentation", [])

        if missing:
            for item in missing:
                st.error(item)
        else:
            st.success("No major missing documentation found.")

    with col3:
        st.markdown("### Recommended Fixes")
        fixes = result.get("recommended_fixes", [])

        if fixes:
            for item in fixes:
                st.info(item)
        else:
            st.success("No major fixes required.")

    st.markdown("### Retrieved Policy Sources")

    sources = result.get("retrieved_sources", [])

    if sources:
        for source in sources:
            st.markdown(
                f"""
                <div class="source-box">
                    <b>Source:</b> {source.get("source_file")}<br>
                    <b>Folder:</b> {source.get("source_folder")}<br>
                    <b>Chunk:</b> {source.get("chunk_index")}<br>
                    <b>Type:</b> {source.get("query_label")}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.write("No sources returned.")


def render_pdf_download_button(patient_data, result_data, button_label="Download PDF Report"):
    pdf_bytes = generate_report_pdf_bytes(
        patient_data=patient_data,
        result_data=result_data,
    )

    patient_name_safe = clean_filename(patient_data.get("patient_name", "patient"))
    date_str = datetime.now().strftime("%Y%m%d")

    st.download_button(
        label=button_label,
        data=pdf_bytes,
        file_name=f"PreAuth_Report_{patient_name_safe}_{date_str}.pdf",
        mime="application/pdf",
    )


def initialize_submission_state():
    defaults = {
        "extracted_fields": {},
        "uploaded_pdf_name": "",
        "latest_result": None,
        "latest_patient_data": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_current_submission():
    st.session_state["extracted_fields"] = {}
    st.session_state["uploaded_pdf_name"] = ""
    st.session_state["latest_result"] = None
    st.session_state["latest_patient_data"] = None


# =============================================================================
# Sidebar navigation
# =============================================================================

st.sidebar.title("PreAuth.ai")
st.sidebar.caption("Prior Authorization Copilot")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "New Submission",
        "Patient Lookup",
        "Model Performance",
        "How It Works",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Analysis Options")

st.sidebar.markdown("### Local Mode")
st.sidebar.write(
    "Runs the analysis locally using document extraction, policy retrieval, "
    "and rule-based critical-gap checks."
)

st.sidebar.markdown("### Gemini Live Mode")
st.sidebar.write(
    "Uses Gemini with retrieved payer-policy context for deeper AI-based review."
)


# =============================================================================
# Dashboard page
# =============================================================================

if page == "Dashboard":
    st.markdown('<div class="title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Overview of prior authorization submissions.</div>',
        unsafe_allow_html=True,
    )

    records = load_records()

    all_submissions = []

    for member_id, patient in records.items():
        for submission in patient.get("submissions", []):
            all_submissions.append(
                {
                    "member_id": member_id,
                    "patient_name": patient.get("patient_name", "Unknown"),
                    **submission,
                }
            )

    high_count = sum(1 for s in all_submissions if s.get("risk_level") == "high")
    medium_count = sum(1 for s in all_submissions if s.get("risk_level") == "medium")
    low_count = sum(1 for s in all_submissions if s.get("risk_level") == "low")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Patients", len(records))
    col2.metric("High Risk", high_count)
    col3.metric("Medium Risk", medium_count)
    col4.metric("Low Risk", low_count)

    st.markdown("### Recent Submissions")

    if not all_submissions:
        st.info("No submissions yet. Go to New Submission to run your first case.")
    else:
        all_submissions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        for sub in all_submissions[:10]:
            level = sub.get("risk_level", "unknown")
            css_class = risk_class(level)

            st.markdown(
                f"""
                <div class="card">
                    <b>{sub.get("patient_name")}</b> — {sub.get("member_id")}<br>
                    <b>Procedure:</b> {sub.get("procedure")}<br>
                    <b>Date:</b> {sub.get("date")}<br>
                    <b>Risk:</b> <span class="{css_class}">{sub.get("risk_score")} / 100 — {level.upper()}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# New Submission page
# =============================================================================

elif page == "New Submission":
    initialize_submission_state()

    st.markdown('<div class="title">New Submission</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload a prior authorization packet or enter details manually.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "Academic prototype: use only synthetic or de-identified demo records. "
        "Do not upload real patient information."
    )

    # -------------------------------------------------------------------------
    # Step 1: Upload PDF
    # -------------------------------------------------------------------------

    st.markdown("### Step 1 — Upload PDF Packet")

    uploaded_file = st.file_uploader(
        "Upload Prior Authorization Packet",
        type=["pdf"],
        help="Upload a PDF packet. The system will extract fields from the document.",
    )

    col_upload_1, col_upload_2 = st.columns([1, 3])

    with col_upload_1:
        extract_clicked = st.button("Extract Fields from PDF")

    with col_upload_2:
        st.markdown(
            '<div class="small-note">Upload a prior authorization packet, extract the fields, review them, and then run the analysis.</div>',
            unsafe_allow_html=True,
        )

    if extract_clicked:
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Extracting text and fields from PDF..."):
                extracted = extract_fields_from_pdf(uploaded_file)

            st.session_state["extracted_fields"] = extracted
            st.session_state["uploaded_pdf_name"] = uploaded_file.name
            st.session_state["latest_result"] = None
            st.session_state["latest_patient_data"] = None

            st.success("Fields extracted. Please review and edit them below.")

            with st.expander("Show extracted raw text"):
                st.text_area(
                    "Raw extracted text",
                    value=extracted.get("raw_text", ""),
                    height=300,
                )

    extracted = st.session_state.get("extracted_fields", {})

    if st.session_state.get("uploaded_pdf_name"):
        st.info(f"Uploaded file: {st.session_state['uploaded_pdf_name']}")

    # -------------------------------------------------------------------------
    # Step 2: Review/edit details
    # -------------------------------------------------------------------------

    st.markdown("### Step 2 — Review / Edit Details")

    with st.form("new_submission_form"):
        col1, col2 = st.columns(2)

        with col1:
            patient_name = st.text_input(
                "Patient Name",
                value=extracted.get("patient_name", "Jordan M. Ellis"),
            )

            insurance_id = st.text_input(
                "Insurance Member ID",
                value=extracted.get("insurance_id", "92746158300"),
            )

            provider_npi = st.text_input(
                "Provider NPI",
                value=extracted.get("provider_npi", "1686526231"),
            )

        with col2:
            requested_service = st.text_input(
                "Drug / Procedure / Requested Service",
                value=extracted.get(
                    "drug_name",
                    "Right Knee Arthroscopy with Partial Medial Meniscectomy - CPT 29881",
                ),
            )

            diagnosis_code = st.text_input(
                "Diagnosis Code",
                value=extracted.get("diagnosis_code", "M23.221, M25.561"),
            )

        clinical_summary = st.text_area(
            "Clinical Summary",
            height=220,
            value=extracted.get(
                "clinical_summary",
                """Patient has right knee pain with MRI-confirmed medial meniscus tear.
Conservative treatment included physical therapy, NSAIDs, activity modification,
and home exercise program for 8 weeks with persistent pain.
Physical exam shows medial joint line tenderness, limited range of motion,
and positive McMurray test. Symptoms cause difficulty walking and climbing stairs.
Provider recommends arthroscopic partial medial meniscectomy due to failed conservative treatment.""",
            ),
        )

        analysis_mode = st.selectbox(
            "Choose Analysis Mode",
            ["Local Mode", "Gemini Live Mode"],
            index=0,
            help=(
                "Local Mode uses local rule-based analysis. "
                "Gemini Live Mode uses Gemini with retrieved policy context."
            ),
        )

        submitted = st.form_submit_button("Run Prior Authorization Analysis")

    if submitted:
        patient_data = {
            "patient_name": patient_name,
            "insurance_id": insurance_id,
            "provider_npi": provider_npi,
            "drug_name": requested_service,
            "diagnosis_code": diagnosis_code,
            "clinical_summary": clinical_summary,
            "raw_text": extracted.get("raw_text", ""),
        }

        with st.spinner("Retrieving payer policies and analyzing denial risk..."):
            result = analyze_prior_auth(
                patient_data,
                analysis_mode=analysis_mode,
            )

        add_submission(patient_data, result)

        st.session_state["latest_result"] = result
        st.session_state["latest_patient_data"] = patient_data

        st.success("Analysis complete.")

    # -------------------------------------------------------------------------
    # Step 3: Display result and PDF download
    # -------------------------------------------------------------------------

    if st.session_state.get("latest_result"):
        st.markdown("### Step 3 — Results")

        latest_patient_data = st.session_state.get("latest_patient_data", {})
        latest_result = st.session_state.get("latest_result", {})

        render_result(latest_result)

        st.markdown("### Export")
        render_pdf_download_button(
            patient_data=latest_patient_data,
            result_data=latest_result,
            button_label="Download PDF Report",
        )

    st.markdown("---")

    if st.button("Clear Current Submission"):
        clear_current_submission()
        st.rerun()


# =============================================================================
# Patient Lookup page
# =============================================================================

elif page == "Patient Lookup":
    st.markdown('<div class="title">Patient Lookup</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Search previous submissions by insurance member ID.</div>',
        unsafe_allow_html=True,
    )

    records = load_records()

    search_id = st.text_input("Enter Insurance Member ID")

    if search_id:
        if search_id in records:
            patient = records[search_id]

            st.markdown(
                f"""
                <div class="card">
                    <h2>{patient.get("patient_name")}</h2>
                    <p><b>Member ID:</b> {search_id}</p>
                    <p><b>Total submissions:</b> {len(patient.get("submissions", []))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            submissions = sorted(
                patient.get("submissions", []),
                key=lambda x: x.get("timestamp", ""),
                reverse=True,
            )

            for sub in submissions:
                result_data = sub.get("result", {})
                patient_data = {
                    "patient_name": patient.get("patient_name", ""),
                    "insurance_id": search_id,
                    "provider_npi": "",
                    "drug_name": sub.get("procedure", ""),
                    "diagnosis_code": "",
                    "clinical_summary": "",
                }

                with st.expander(
                    f"{sub.get('date')} — {sub.get('procedure')} — {sub.get('risk_score')} / 100 {sub.get('risk_level', '').upper()}"
                ):
                    render_result(result_data)

                    st.markdown("#### Export")
                    render_pdf_download_button(
                        patient_data=patient_data,
                        result_data=result_data,
                        button_label="Download This Report",
                    )

        else:
            st.warning("No patient found for this member ID.")
    else:
        st.info("Enter a member ID to search.")

        if records:
            st.markdown("### All Patients")

            for member_id, patient in records.items():
                st.markdown(
                    f"""
                    <div class="card">
                        <b>{patient.get("patient_name")}</b><br>
                        Member ID: {member_id}<br>
                        Submissions: {len(patient.get("submissions", []))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# Model Performance page
# =============================================================================

elif page == "Model Performance":
    st.markdown("## Model Performance")

    st.markdown(
        "This page summarizes the evaluation of the prior authorization risk analysis system "
        "using 40 synthetic prior authorization records with varying levels of documentation completeness."
    )

    st.markdown("### Evaluation Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Synthetic Records", "40")

    with col2:
        st.metric("Evaluation Type", "Synthetic Benchmark")

    with col3:
        st.metric("Output Type", "Risk Score + Risk Level")

    with col4:
        st.metric("Fallback Behavior", "Implemented")

    st.markdown("### What Was Evaluated")

    st.markdown(
        """
        The evaluation tested whether the system could distinguish between:

        - **Complete packets** with strong medical-necessity support
        - **Medium-risk packets** with partial, ambiguous, or incomplete documentation
        - **High-risk packets** with major missing documentation, outdated evidence, coding issues, or administrative inconsistencies
        """
    )

    st.markdown("### Evaluation Criteria")

    evaluation_criteria = [
        {
            "Evaluation Area": "Risk Classification",
            "What It Measures": "Whether the system assigns low, medium, or high denial risk appropriately.",
            "Business Relevance": "Helps prior authorization teams prioritize packets that need review before submission.",
        },
        {
            "Evaluation Area": "Missing Documentation Detection",
            "What It Measures": "Whether the system identifies missing or incomplete clinical/administrative evidence.",
            "Business Relevance": "Reduces preventable denials caused by incomplete documentation.",
        },
        {
            "Evaluation Area": "Recommended Fix Quality",
            "What It Measures": "Whether the recommendations are specific, actionable, and relevant to the packet.",
            "Business Relevance": "Helps staff correct issues faster before payer submission.",
        },
        {
            "Evaluation Area": "Policy Retrieval Relevance",
            "What It Measures": "Whether retrieved payer-policy excerpts are related to the requested service and documentation need.",
            "Business Relevance": "Improves trust and explainability for reviewers.",
        },
        {
            "Evaluation Area": "Fallback Reliability",
            "What It Measures": "Whether the app still returns an analysis when the live LLM API is unavailable.",
            "Business Relevance": "Keeps the workflow usable during API quota, outage, or connectivity issues.",
        },
    ]

    st.dataframe(evaluation_criteria, use_container_width=True)

    st.markdown("### Business Interpretation")

    st.info(
        "The synthetic benchmark is designed to test whether the system can support prior authorization staff "
        "by identifying documentation risk before payer submission. In a real workflow, this could reduce manual "
        "review time, improve first-pass submission quality, and lower avoidable denial risk."
    )

    st.markdown("### Current Evaluation Limitations")

    st.warning(
        "The current benchmark uses synthetic academic records rather than real payer outcomes. "
        "Before production use, the system should be validated on a larger de-identified dataset with actual "
        "authorization outcomes, denial reasons, reviewer feedback, and payer-specific decisions."
    )

    st.markdown("### Next Evaluation Improvements")

    st.markdown(
        """
        Future evaluation should include:

        - Precision and recall for missing-documentation detection
        - False-positive review of incorrectly flagged gaps
        - Human reviewer scoring of recommendation quality
        - Hallucination checks against the uploaded packet and retrieved policy excerpts
        - Comparison of Local Mode, Gemini Live Mode, and future Claude Live Mode
        - Business KPI tracking such as review time saved, denial-prevention rate, and first-pass approval improvement
        """
    )


# =============================================================================
# How It Works page
# =============================================================================

elif page == "How It Works":
    st.markdown("## How It Works")

    st.markdown(
        "PreAuth.ai helps prior authorization teams review documentation packets before submission. "
        "The system checks whether the packet appears complete, retrieves relevant payer-policy context, "
        "and highlights denial-risk factors that may need attention."
    )

    st.markdown("### Workflow")

    st.markdown(
        """
        1. **Upload prior authorization packet**  
           The user uploads a synthetic or de-identified prior authorization PDF packet.

        2. **Extract key fields**  
           The app extracts patient, insurance, provider, diagnosis, procedure, and clinical summary details.

        3. **Retrieve payer-policy context**  
           The system searches the policy knowledge base and retrieves relevant policy/documentation excerpts.

        4. **Analyze documentation risk**  
           The analysis engine checks the packet for missing, incomplete, outdated, or inconsistent documentation.

        5. **Generate business-facing output**  
           The app displays a risk score, risk level, denial reasons, missing documentation, recommended fixes, and retrieved policy sources.

        6. **Export report**  
           The user can download a PDF summary report for review or presentation.
        """
    )

    st.markdown("### Analysis Modes")

    st.markdown(
        """
        **Local Mode**  
        Uses policy retrieval and rule-based critical-gap checks.

        **Gemini Live Mode**  
        Uses Gemini with retrieved payer-policy context for deeper AI-based review when an API key and quota are available.
        """
    )

    st.markdown("### Guardrails")

    st.markdown(
        """
        - The system is designed for prior authorization documentation review support, not autonomous decision-making.
        - The model is instructed to use only the uploaded packet content and retrieved policy excerpts.
        - Retrieved policy sources are displayed to improve transparency.
        - If the live AI mode is unavailable, the app can fall back to local analysis.
        - Users should review outputs before making any operational or submission decision.
        """
    )

    st.markdown("### Important Notes")

    st.warning(
        "This application is an academic prototype and should be used only with synthetic or de-identified demo records. "
        "It is not intended for real patient-care decisions, payer submissions, or medical/legal advice."
    )

    st.info(
        "The system is designed as decision support for prior authorization documentation review. "
        "Final review should always remain with qualified healthcare, administrative, or compliance staff."
    )
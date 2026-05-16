import json
import re
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
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
EVALUATION_FILE = DATA_DIR / "model_evaluation_records.json"


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
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Data functions — prior authorization records
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
# Data functions — evaluation records
# =============================================================================

def load_evaluation_records():
    if EVALUATION_FILE.exists():
        with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_evaluation_records(records):
    with open(EVALUATION_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def add_evaluation_record(record):
    records = load_evaluation_records()
    records.append(record)
    save_evaluation_records(records)


def clear_evaluation_records():
    save_evaluation_records([])


# =============================================================================
# Evaluation helper functions
# =============================================================================

def normalize_risk_label(label):
    label = str(label).lower().strip()

    if label.startswith("low"):
        return "low"
    if label.startswith("medium"):
        return "medium"
    if label.startswith("high"):
        return "high"

    return "unknown"


def build_confusion_matrix(records):
    labels = ["low", "medium", "high"]

    matrix = pd.DataFrame(
        0,
        index=labels,
        columns=labels,
    )

    for record in records:
        expected = normalize_risk_label(record.get("expected_risk", ""))
        predicted = normalize_risk_label(record.get("predicted_risk", ""))

        if expected in labels and predicted in labels:
            matrix.loc[expected, predicted] += 1

    return matrix


def calculate_precision_recall(matrix):
    labels = ["low", "medium", "high"]

    rows = []
    total_actual = matrix.values.sum()

    weighted_precision_sum = 0
    weighted_recall_sum = 0

    for label in labels:
        true_positive = matrix.loc[label, label]
        predicted_total = matrix[label].sum()
        actual_total = matrix.loc[label].sum()

        precision = true_positive / predicted_total if predicted_total > 0 else 0
        recall = true_positive / actual_total if actual_total > 0 else 0

        rows.append(
            {
                "Risk Class": label.upper(),
                "Actual Count": int(actual_total),
                "Predicted Count": int(predicted_total),
                "True Positives": int(true_positive),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
            }
        )

        if total_actual > 0:
            class_weight = actual_total / total_actual
            weighted_precision_sum += precision * class_weight
            weighted_recall_sum += recall * class_weight

    metrics_df = pd.DataFrame(rows)

    return metrics_df, round(weighted_precision_sum, 3), round(weighted_recall_sum, 3)


def calculate_business_metrics(
    weekly_prior_auths,
    current_minutes_per_pa,
    current_denial_rate,
    current_adverse_event_rate,
    weighted_precision,
    weighted_recall,
    max_time_reduction_pct,
    preventable_denial_pct,
    preventable_adverse_event_pct,
):
    max_time_reduction = max_time_reduction_pct / 100
    preventable_denial = preventable_denial_pct / 100
    preventable_adverse = preventable_adverse_event_pct / 100

    time_saved_per_pa = current_minutes_per_pa * max_time_reduction * weighted_precision
    weekly_time_saved_hours = (weekly_prior_auths * time_saved_per_pa) / 60

    denial_rate_reduction = current_denial_rate * preventable_denial * weighted_recall
    projected_denial_rate = max(current_denial_rate - denial_rate_reduction, 0)

    adverse_event_reduction = current_adverse_event_rate * preventable_adverse * weighted_recall
    projected_adverse_event_rate = max(current_adverse_event_rate - adverse_event_reduction, 0)

    return {
        "time_saved_per_pa": round(time_saved_per_pa, 2),
        "weekly_time_saved_hours": round(weekly_time_saved_hours, 2),
        "denial_rate_reduction": round(denial_rate_reduction, 4),
        "projected_denial_rate": round(projected_denial_rate, 4),
        "adverse_event_reduction": round(adverse_event_reduction, 4),
        "projected_adverse_event_rate": round(projected_adverse_event_rate, 4),
    }


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

    if analysis_mode:
        st.caption(f"Backend mode used: {analysis_mode}")

    if analysis_mode == "fallback_local_rules":
        if result.get("groq_error"):
            st.warning(
                "Groq AI Review Mode was unavailable, so the app used fallback logic. "
                "Check GROQ_API_KEY, rate limits, or model availability."
            )

        if result.get("gemini_error"):
            st.warning(
                "Gemini Live Mode was also unavailable or not configured."
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
        "Test Evaluation",
        "Model Performance",
        "How It Works",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Analysis Options")

st.sidebar.markdown("### Groq AI Review Mode")
st.sidebar.write(
    "Default AI review mode. Uses Llama 3.3 70B through Groq with uploaded packet text and retrieved payer-policy context."
)

st.sidebar.markdown("### Gemini Live Mode")
st.sidebar.write(
    "Backup AI review mode. Uses Gemini with uploaded packet text and retrieved payer-policy context."
)

st.sidebar.markdown("### Local Fallback Mode")
st.sidebar.write(
    "Backup mode only. Used when live AI APIs are unavailable."
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
            ["Groq AI Review Mode", "Gemini Live Mode", "Local Fallback Mode"],
            index=0,
            help=(
                "Groq AI Review Mode is the default AI reviewer. "
                "Gemini Live Mode is an alternate AI reviewer. "
                "Local Fallback Mode is only a backup if APIs are unavailable."
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
# Test Evaluation page
# =============================================================================

elif page == "Test Evaluation":
    st.markdown("## Test Evaluation")

    st.markdown(
        "Use this page to evaluate model performance against synthetic test records. "
        "Upload each test PDF, enter the expected risk category, choose an analysis mode, "
        "and save the model prediction. The app will automatically build a 3×3 confusion matrix, "
        "precision/recall scores, and business impact estimates."
    )

    st.warning(
        "Use synthetic or de-identified academic test records only. Do not upload real patient data."
    )

    st.markdown("### Step 1 — Upload Test Record and Expected Label")

    uploaded_eval_file = st.file_uploader(
        "Upload synthetic test PDF",
        type=["pdf"],
        key="evaluation_pdf_upload",
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        expected_risk = st.selectbox(
            "Expected Risk Category",
            ["low", "medium", "high"],
            index=0,
        )

    with col_b:
        eval_analysis_mode = st.selectbox(
            "Model / Analysis Mode",
            ["Groq AI Review Mode", "Gemini Live Mode", "Local Fallback Mode"],
            index=0,
        )

    with col_c:
        test_record_name = st.text_input(
            "Test Record Name / ID",
            value="",
            placeholder="Example: Record 01",
        )

    run_eval = st.button("Run Evaluation on This Record")

    if run_eval:
        if uploaded_eval_file is None:
            st.warning("Please upload a synthetic PDF test record first.")
        else:
            with st.spinner("Extracting fields and running model evaluation..."):
                extracted = extract_fields_from_pdf(uploaded_eval_file)

                patient_data = {
                    "patient_name": extracted.get("patient_name", ""),
                    "insurance_id": extracted.get("insurance_id", ""),
                    "provider_npi": extracted.get("provider_npi", ""),
                    "drug_name": extracted.get("drug_name", ""),
                    "diagnosis_code": extracted.get("diagnosis_code", ""),
                    "clinical_summary": extracted.get("clinical_summary", ""),
                    "raw_text": extracted.get("raw_text", ""),
                }

                result = analyze_prior_auth(
                    patient_data,
                    analysis_mode=eval_analysis_mode,
                )

            predicted_risk = normalize_risk_label(result.get("risk_level", "unknown"))
            expected_risk_clean = normalize_risk_label(expected_risk)

            evaluation_record = {
                "timestamp": datetime.now().isoformat(),
                "record_name": test_record_name or uploaded_eval_file.name,
                "file_name": uploaded_eval_file.name,
                "analysis_mode": eval_analysis_mode,
                "backend_mode": result.get("analysis_mode", ""),
                "expected_risk": expected_risk_clean,
                "predicted_risk": predicted_risk,
                "risk_score": result.get("risk_score", 0),
                "match": expected_risk_clean == predicted_risk,
                "denial_reasons": result.get("denial_reasons", []),
                "missing_documentation": result.get("missing_documentation", []),
                "recommended_fixes": result.get("recommended_fixes", []),
                "retrieved_sources": result.get("retrieved_sources", []),
            }

            add_evaluation_record(evaluation_record)

            st.success("Evaluation record saved.")

            st.markdown("### Model Output for This Record")
            render_result(result)

    st.markdown("---")
    st.markdown("## Evaluation Results")

    evaluation_records = load_evaluation_records()

    if not evaluation_records:
        st.info("No evaluation records saved yet. Upload a test PDF above to begin.")
    else:
        eval_table = []

        for record in evaluation_records:
            eval_table.append(
                {
                    "Record": record.get("record_name", ""),
                    "File": record.get("file_name", ""),
                    "Mode": record.get("analysis_mode", ""),
                    "Backend": record.get("backend_mode", ""),
                    "Expected": record.get("expected_risk", ""),
                    "Predicted": record.get("predicted_risk", ""),
                    "Risk Score": record.get("risk_score", ""),
                    "Match": "Yes" if record.get("match") else "No",
                }
            )

        eval_df = pd.DataFrame(eval_table)

        st.markdown("### Saved Evaluation Records")
        st.dataframe(eval_df, use_container_width=True)

        confusion_matrix = build_confusion_matrix(evaluation_records)

        st.markdown("### 3×3 Confusion Matrix")
        st.dataframe(confusion_matrix, use_container_width=True)

        st.caption(
            "Rows represent the expected risk category. Columns represent the model-generated risk category."
        )

        metrics_df, weighted_precision, weighted_recall = calculate_precision_recall(
            confusion_matrix
        )

        st.markdown("### Precision and Recall")
        st.dataframe(metrics_df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Weighted Precision", weighted_precision)

        with col2:
            st.metric("Weighted Recall", weighted_recall)

        st.markdown("### Business Impact Assumptions")

        st.markdown(
            "These fields translate model performance into estimated business impact. "
            "The estimates are directional and depend on the assumptions entered below."
        )

        col1, col2 = st.columns(2)

        with col1:
            weekly_prior_auths = st.number_input(
                "Prior authorizations completed per week",
                min_value=1,
                value=100,
                step=10,
            )

            current_minutes_per_pa = st.number_input(
                "Current staff time per prior authorization review (minutes)",
                min_value=1,
                value=25,
                step=1,
            )

            current_denial_rate_pct = st.number_input(
                "Current denial rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=18.0,
                step=0.5,
            )

        with col2:
            current_adverse_event_rate_pct = st.number_input(
                "Current adverse patient event / delay rate due to PA review (%)",
                min_value=0.0,
                max_value=100.0,
                value=3.0,
                step=0.1,
            )

            max_time_reduction_pct = st.number_input(
                "Assumed maximum time reduction when model is reliable (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
            )

            preventable_denial_pct = st.number_input(
                "Share of denials assumed preventable through better documentation (%)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=1.0,
            )

            preventable_adverse_event_pct = st.number_input(
                "Share of adverse events / delays assumed preventable through better review (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
            )

        business_metrics = calculate_business_metrics(
            weekly_prior_auths=weekly_prior_auths,
            current_minutes_per_pa=current_minutes_per_pa,
            current_denial_rate=current_denial_rate_pct / 100,
            current_adverse_event_rate=current_adverse_event_rate_pct / 100,
            weighted_precision=weighted_precision,
            weighted_recall=weighted_recall,
            max_time_reduction_pct=max_time_reduction_pct,
            preventable_denial_pct=preventable_denial_pct,
            preventable_adverse_event_pct=preventable_adverse_event_pct,
        )

        st.markdown("### Estimated Business Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Estimated Time Saved per PA",
                f"{business_metrics['time_saved_per_pa']} min",
            )

        with col2:
            st.metric(
                "Estimated Weekly Time Saved",
                f"{business_metrics['weekly_time_saved_hours']} hrs",
            )

        with col3:
            st.metric(
                "Projected Denial Rate",
                f"{business_metrics['projected_denial_rate'] * 100:.1f}%",
                delta=f"-{business_metrics['denial_rate_reduction'] * 100:.1f}%",
            )

        col4, col5 = st.columns(2)

        with col4:
            st.metric(
                "Projected Adverse Event / Delay Rate",
                f"{business_metrics['projected_adverse_event_rate'] * 100:.2f}%",
                delta=f"-{business_metrics['adverse_event_reduction'] * 100:.2f}%",
            )

        with col5:
            total_records = len(evaluation_records)
            correct_records = sum(1 for r in evaluation_records if r.get("match"))
            accuracy = correct_records / total_records if total_records > 0 else 0

            st.metric(
                "Risk Label Accuracy",
                f"{accuracy * 100:.1f}%",
            )

        st.info(
            "Business estimates are directional. Weighted precision is used to estimate review-time efficiency, "
            "while weighted recall is used to estimate potential denial and patient-delay reduction because recall "
            "measures how well the model catches true risk cases."
        )

        st.markdown("### Export Evaluation Data")

        csv_data = eval_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Evaluation Records CSV",
            data=csv_data,
            file_name="model_evaluation_records.csv",
            mime="text/csv",
        )

        st.markdown("---")

        if st.button("Clear All Evaluation Records"):
            clear_evaluation_records()
            st.success("Evaluation records cleared.")
            st.rerun()


# =============================================================================
# Model Performance page
# =============================================================================

elif page == "Model Performance":
    st.markdown("## Model Performance")

    evaluation_records = load_evaluation_records()

    st.markdown(
        "This page summarizes the current model evaluation results from the Test Evaluation page."
    )

    if evaluation_records:
        confusion_matrix = build_confusion_matrix(evaluation_records)
        metrics_df, weighted_precision, weighted_recall = calculate_precision_recall(confusion_matrix)

        total_records = len(evaluation_records)
        correct_records = sum(1 for r in evaluation_records if r.get("match"))
        accuracy = correct_records / total_records if total_records > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Evaluation Records", total_records)

        with col2:
            st.metric("Risk Label Accuracy", f"{accuracy * 100:.1f}%")

        with col3:
            st.metric("Weighted Precision", weighted_precision)

        with col4:
            st.metric("Weighted Recall", weighted_recall)

        st.markdown("### Current Confusion Matrix")
        st.dataframe(confusion_matrix, use_container_width=True)

        st.markdown("### Class-Level Precision and Recall")
        st.dataframe(metrics_df, use_container_width=True)

    else:
        st.info(
            "No evaluation records have been saved yet. Go to the Test Evaluation page, upload synthetic records, "
            "enter expected risk labels, and run the model to generate performance metrics."
        )

    st.markdown("### What the Evaluation Measures")

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
        "The evaluation workflow tests whether the system can support prior authorization staff by identifying "
        "documentation risk before payer submission. In a real workflow, this could reduce manual review time, "
        "improve first-pass submission quality, and lower avoidable denial risk."
    )

    st.markdown("### Current Evaluation Limitations")

    st.warning(
        "The current benchmark uses synthetic academic records rather than real payer outcomes. "
        "Before production use, the system should be validated on a larger de-identified dataset with actual "
        "authorization outcomes, denial reasons, reviewer feedback, and payer-specific decisions."
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
           The AI review engine checks the packet for missing, incomplete, outdated, or inconsistent documentation.

        5. **Generate business-facing output**  
           The app displays a risk score, risk level, denial reasons, missing documentation, recommended fixes, and retrieved policy sources.

        6. **Export report**  
           The user can download a PDF summary report for review or presentation.
        """
    )

    st.markdown("### Analysis Modes")

    st.markdown(
        """
        **Groq AI Review Mode**  
        Default AI review mode using uploaded packet text and retrieved policy context.

        **Gemini Live Mode**  
        Backup AI review mode using uploaded packet text and retrieved policy context.

        **Local Fallback Mode**  
        Backup mode used only when live AI APIs are unavailable.
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

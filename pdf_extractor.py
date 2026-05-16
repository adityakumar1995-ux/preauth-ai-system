import re
import fitz  # PyMuPDF


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extracts raw text from an uploaded PDF file.

    uploaded_file comes from Streamlit's file_uploader.
    """
    pdf_bytes = uploaded_file.getvalue()
    text_parts = []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()

            if page_text.strip():
                text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")

        doc.close()

    except Exception as e:
        return f"PDF extraction error: {e}"

    return "\n".join(text_parts)


def clean_value(value: str) -> str:
    """
    Cleans extracted field values.
    """
    if not value:
        return ""

    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = value.replace(":", "").strip()

    return value


def find_first_match(patterns, text):
    """
    Tries multiple regex patterns and returns the first match.
    """
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

        if match:
            return clean_value(match.group(1))

    return ""


def extract_fields_from_text(text: str) -> dict:
    """
    Extracts common prior authorization fields using simple regex rules.

    This is not perfect AI extraction yet.
    It is a free/local first version.
    The user can review and edit extracted fields before analysis.
    """

    patient_name = find_first_match(
        [
            r"Patient Name\s*[:\-]?\s*([A-Za-z ,.'-]+)",
            r"Name\s*[:\-]?\s*([A-Za-z ,.'-]+)",
        ],
        text,
    )

    insurance_id = find_first_match(
        [
            r"Insurance Member ID\s*[:\-]?\s*([A-Za-z0-9\-]+)",
            r"Member ID\s*[:\-]?\s*([A-Za-z0-9\-]+)",
            r"Policy ID\s*[:\-]?\s*([A-Za-z0-9\-]+)",
            r"Subscriber ID\s*[:\-]?\s*([A-Za-z0-9\-]+)",
        ],
        text,
    )

    diagnosis_code = find_first_match(
        [
            r"Diagnosis Code\s*\(ICD-10\)\s*[:\-]?\s*([A-Z][0-9][0-9A-Z. ,\-]+)",
            r"ICD-10\s*[:\-]?\s*([A-Z][0-9][0-9A-Z. ,\-]+)",
            r"Diagnosis\s*[:\-]?\s*([A-Z][0-9][0-9A-Z. ,\-]+)",
        ],
        text,
    )

    provider_npi = find_first_match(
        [
            r"Provider NPI\s*[:\-]?\s*([0-9]{10})",
            r"NPI\s*[:\-]?\s*([0-9]{10})",
        ],
        text,
    )

    cpt_code = find_first_match(
        [
            r"CPT\s*[:\-]?\s*([0-9]{5})",
            r"CPT Code\s*[:\-]?\s*([0-9]{5})",
            r"Procedure Code\s*[:\-]?\s*([0-9]{5})",
        ],
        text,
    )

    requested_service = find_first_match(
        [
            r"Drug or Procedure\s*[:\-]?\s*(.+)",
            r"Requested Service\s*[:\-]?\s*(.+)",
            r"Procedure\s*[:\-]?\s*(.+)",
            r"Service Requested\s*[:\-]?\s*(.+)",
        ],
        text,
    )

    if requested_service and cpt_code and cpt_code not in requested_service:
        requested_service = f"{requested_service} - CPT {cpt_code}"

    if not requested_service and cpt_code:
        requested_service = f"CPT {cpt_code}"

    clinical_summary = build_simple_clinical_summary(text)

    return {
        "patient_name": patient_name,
        "insurance_id": insurance_id,
        "drug_name": requested_service,
        "diagnosis_code": diagnosis_code,
        "provider_npi": provider_npi,
        "clinical_summary": clinical_summary,
        "raw_text": text,
    }


def build_simple_clinical_summary(text: str) -> str:
    """
    Creates a rough clinical summary from useful clinical sentences.

    This version removes administrative/header lines first, such as:
    Patient Name, Member ID, Diagnosis Code, Provider NPI, CPT Code, etc.
    """

    # Split into lines first so we can remove full admin lines cleanly
    lines = text.splitlines()

    admin_keywords = [
        "patient name",
        "insurance member id",
        "member id",
        "policy id",
        "subscriber id",
        "drug or procedure",
        "requested service",
        "service requested",
        "procedure:",
        "diagnosis code",
        "icd-10",
        "provider npi",
        "npi:",
        "cpt code",
        "cpt:",
        "group number",
        "insurance plan",
        "provider name",
        "facility name",
        "date of service",
    ]

    clinical_lines = []

    for line in lines:
        clean_line = line.strip()
        lower_line = clean_line.lower()

        if not clean_line:
            continue

        # Skip administrative/header lines
        if any(keyword in lower_line for keyword in admin_keywords):
            continue

        clinical_lines.append(clean_line)

    cleaned_text = " ".join(clinical_lines)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    keywords = [
        "pain",
        "mri",
        "x-ray",
        "xray",
        "imaging",
        "physical therapy",
        "conservative",
        "functional",
        "activities of daily living",
        "range of motion",
        "tenderness",
        "meniscus",
        "tear",
        "failed",
        "treatment",
        "provider recommends",
        "medical necessity",
        "positive",
        "mcmurray",
        "limitation",
        "walking",
        "stairs",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)

    useful_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        sentence_lower = sentence.lower()

        if len(sentence) < 20:
            continue

        if any(keyword in sentence_lower for keyword in keywords):
            useful_sentences.append(sentence)

        if len(useful_sentences) >= 6:
            break

    if useful_sentences:
        return " ".join(useful_sentences)

    return ""


def extract_fields_from_pdf(uploaded_file) -> dict:
    """
    Main function used by Streamlit.

    PDF → text → fields
    """
    text = extract_text_from_pdf(uploaded_file)
    fields = extract_fields_from_text(text)
    return fields
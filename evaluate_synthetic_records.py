from pathlib import Path
import csv
import re

from pdf_extractor import extract_fields_from_text
from analysis_engine import analyze_prior_auth


PROJECT_ROOT = Path(__file__).parent
EVAL_DIR = PROJECT_ROOT / "evaluation_records"
OUTPUT_CSV = PROJECT_ROOT / "evaluation_results.csv"


EXPECTED_RISK_BY_RECORD = {
    "01": "low",
    "02": "low",
    "03": "low",
    "04": "medium",
    "05": "high",
    "06": "high",
    "07": "medium",
    "08": "high",
    "09": "high",
    "10": "high",
}


EXPECTED_SCORE_RANGE = {
    "low": (0, 35),
    "medium": (36, 69),
    "high": (70, 100),
}


def read_pdf_text_from_path(pdf_path: Path) -> str:
    import fitz

    text_parts = []

    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()

        if text.strip():
            text_parts.append(f"\n--- Page {page_num} ---\n{text}")

    doc.close()

    return "\n".join(text_parts)


def get_record_number(filename: str) -> str:
    match = re.match(r"(\d{2})", filename)

    if match:
        return match.group(1)

    return "unknown"


def score_in_expected_range(score: int, expected_risk: str) -> bool:
    low, high = EXPECTED_SCORE_RANGE[expected_risk]
    return low <= score <= high


def classify_match(predicted_risk: str, expected_risk: str, score: int) -> str:
    predicted_risk = predicted_risk.lower()
    expected_risk = expected_risk.lower()

    if predicted_risk == expected_risk and score_in_expected_range(score, expected_risk):
        return "exact_match"

    if predicted_risk == expected_risk:
        return "risk_label_match_score_range_off"

    return "mismatch"


def evaluate_one_pdf(pdf_path: Path, demo_mode: bool = True) -> dict:
    record_number = get_record_number(pdf_path.name)
    expected_risk = EXPECTED_RISK_BY_RECORD.get(record_number, "unknown")

    raw_text = read_pdf_text_from_path(pdf_path)
    extracted = extract_fields_from_text(raw_text)

    patient_data = {
        "patient_name": extracted.get("patient_name", ""),
        "insurance_id": extracted.get("insurance_id", ""),
        "provider_npi": extracted.get("provider_npi", ""),
        "drug_name": extracted.get("drug_name", ""),
        "diagnosis_code": extracted.get("diagnosis_code", ""),
        "clinical_summary": extracted.get("clinical_summary", ""),
        "raw_text": raw_text,
    }

    result = analyze_prior_auth(
        patient_data=patient_data,
        demo_mode=demo_mode,
    )

    score = int(result.get("risk_score", 0))
    predicted_risk = result.get("risk_level", "unknown").lower()

    return {
        "record_number": record_number,
        "filename": pdf_path.name,
        "expected_risk": expected_risk,
        "predicted_risk": predicted_risk,
        "risk_score": score,
        "match_status": classify_match(predicted_risk, expected_risk, score)
        if expected_risk != "unknown"
        else "unknown_expected",
        "analysis_mode": result.get("analysis_mode", ""),
        "gemini_error": result.get("gemini_error", ""),
        "patient_name": patient_data.get("patient_name", ""),
        "requested_service": patient_data.get("drug_name", ""),
        "diagnosis_code": patient_data.get("diagnosis_code", ""),
        "missing_documentation_count": len(result.get("missing_documentation", [])),
        "critical_gap_count": len(result.get("critical_gaps", [])),
        "missing_documentation": " | ".join(result.get("missing_documentation", [])),
        "critical_gaps": " | ".join(
            [gap.get("label", "") for gap in result.get("critical_gaps", [])]
        ),
        "denial_reasons": " | ".join(result.get("denial_reasons", [])),
        "recommended_fixes": " | ".join(result.get("recommended_fixes", [])),
    }


def main():
    print("=" * 80)
    print("PreAuth.ai Synthetic Record Evaluation")
    print("=" * 80)

    if not EVAL_DIR.exists():
        print(f"Folder not found: {EVAL_DIR}")
        print("Create evaluation_records/ and place the 10 synthetic PDFs inside it.")
        return

    pdfs = sorted(EVAL_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in: {EVAL_DIR}")
        return

    print(f"Found {len(pdfs)} PDF records.")

    # Change this:
    # True  = rule-based demo mode
    # False = Gemini live mode
    DEMO_MODE = False

    rows = []

    for pdf_path in pdfs:
        print(f"\nEvaluating: {pdf_path.name}")

        try:
            row = evaluate_one_pdf(pdf_path, demo_mode=DEMO_MODE)
            rows.append(row)

            print(
                f"  Expected: {row['expected_risk']} | "
                f"Predicted: {row['predicted_risk']} | "
                f"Score: {row['risk_score']} | "
                f"Status: {row['match_status']} | "
                f"Mode: {row['analysis_mode']} | "
                f"Critical gaps: {row['critical_gap_count']}"
            )

            if row.get("gemini_error"):
                print(f"  Gemini error: {row['gemini_error']}")

        except Exception as e:
            print(f"  ERROR: {e}")

            rows.append({
                "record_number": get_record_number(pdf_path.name),
                "filename": pdf_path.name,
                "expected_risk": EXPECTED_RISK_BY_RECORD.get(
                    get_record_number(pdf_path.name), "unknown"
                ),
                "predicted_risk": "error",
                "risk_score": "",
                "match_status": "error",
                "analysis_mode": "",
                "gemini_error": str(e),
                "patient_name": "",
                "requested_service": "",
                "diagnosis_code": "",
                "missing_documentation_count": "",
                "critical_gap_count": "",
                "missing_documentation": "",
                "critical_gaps": "",
                "denial_reasons": str(e),
                "recommended_fixes": "",
            })

    if rows:
        fieldnames = list(rows[0].keys())

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 80)
    print("Evaluation complete")
    print(f"Results saved to: {OUTPUT_CSV}")
    print("=" * 80)

    total = len(rows)
    exact = sum(1 for r in rows if r["match_status"] == "exact_match")
    label_match = sum(
        1
        for r in rows
        if r["match_status"] in ["exact_match", "risk_label_match_score_range_off"]
    )

    print(f"Total records evaluated : {total}")
    print(f"Exact matches           : {exact}/{total}")
    print(f"Risk-label matches      : {label_match}/{total}")

    print("\nSummary:")
    for r in rows:
        print(
            f"{r['record_number']} | "
            f"Expected={r['expected_risk']} | "
            f"Predicted={r['predicted_risk']} | "
            f"Score={r['risk_score']} | "
            f"Mode={r['analysis_mode']} | "
            f"CriticalGaps={r['critical_gap_count']} | "
            f"{r['match_status']}"
        )

        if r.get("gemini_error"):
            print(f"   Gemini error: {r['gemini_error']}")


if __name__ == "__main__":
    main()
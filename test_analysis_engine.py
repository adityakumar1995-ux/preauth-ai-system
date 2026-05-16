from analysis_engine import analyze_prior_auth, pretty_print_result


patient_data = {
    "patient_name": "Jordan M. Ellis",
    "insurance_id": "92746158300",
    "drug_name": "Right Knee Arthroscopy with Partial Medial Meniscectomy - CPT 29881",
    "diagnosis_code": "M23.221, M25.561",
    "provider_npi": "1686526231",
    "clinical_summary": """
    Patient has right knee pain with MRI-confirmed medial meniscus tear.
    Conservative treatment included physical therapy, NSAIDs, activity modification,
    and home exercise program for 8 weeks with persistent pain.
    Physical exam shows medial joint line tenderness, limited range of motion,
    and positive McMurray test. Symptoms cause difficulty walking and climbing stairs.
    Provider recommends arthroscopic partial medial meniscectomy due to failed conservative treatment.
    """
}


result = analyze_prior_auth(patient_data, demo_mode=True)
pretty_print_result(result)
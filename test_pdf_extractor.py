from pathlib import Path
from pdf_extractor import extract_fields_from_text


sample_text = """
Patient Name: Jordan M. Ellis
Insurance Member ID: 92746158300
Drug or Procedure: Right Knee Arthroscopy with Partial Medial Meniscectomy
Diagnosis Code (ICD-10): M23.221, M25.561
Provider NPI: 1686526231
CPT Code: 29881

Patient has right knee pain with MRI-confirmed medial meniscus tear.
Conservative treatment included physical therapy, NSAIDs, activity modification,
and home exercise program for 8 weeks with persistent pain.
Physical exam shows medial joint line tenderness, limited range of motion,
and positive McMurray test. Symptoms cause difficulty walking and climbing stairs.
Provider recommends arthroscopic partial medial meniscectomy due to failed conservative treatment.
"""

fields = extract_fields_from_text(sample_text)

for key, value in fields.items():
    if key != "raw_text":
        print(f"{key}: {value}")
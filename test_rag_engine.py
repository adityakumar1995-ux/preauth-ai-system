from rag_engine import retrieve_policy_context, format_context_for_prompt


patient_data = {
    "patient_name": "Jordan M. Ellis",
    "insurance_id": "92746158300",
    "drug_name": "Right Knee Arthroscopy with Partial Medial Meniscectomy - CPT 29881",
    "diagnosis_code": "M23.221, M25.561",
    "provider_npi": "1686526231",
    "clinical_summary": """
    Patient has right knee pain, medial meniscus tear, MRI findings,
    failed conservative therapy, and functional limitations.
    """
}


retrieved = retrieve_policy_context(patient_data, n_results=10, final_k=6)

print("=" * 80)
print("TOP RETRIEVED CHUNKS AFTER RERANKING")
print("=" * 80)

for i, chunk in enumerate(retrieved, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Source file   : {chunk['source_file']}")
    print(f"Source folder : {chunk['source_folder']}")
    print(f"Chunk index   : {chunk['chunk_index']}")
    print(f"Distance      : {chunk['distance']}")
    print(f"Keyword boost : {chunk['keyword_boost']}")
    print(f"Query label   : {chunk.get('query_label', 'unknown')}")
    print("-" * 80)
    print(chunk["text"][:1200])
    print("-" * 80)


print("\n\n" + "=" * 80)
print("FORMATTED CONTEXT SAMPLE")
print("=" * 80)

context = format_context_for_prompt(retrieved)
print(context[:3000])
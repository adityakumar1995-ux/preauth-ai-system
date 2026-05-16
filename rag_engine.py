from pathlib import Path
import subprocess
import sys

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


PROJECT_ROOT = Path(__file__).parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

COLLECTION_NAME = "uhc_prior_auth_knowledge"


# =============================================================================
# Vector store bootstrap
# =============================================================================

def vector_store_looks_ready() -> bool:
    """
    Checks whether the local Chroma vector store appears to exist.

    Locally, vector_store/ already exists.
    On Streamlit Cloud, vector_store/ will not be pushed to GitHub,
    so the app needs to rebuild it from knowledge_base/.
    """
    chroma_db_file = VECTOR_STORE_DIR / "chroma.sqlite3"

    return VECTOR_STORE_DIR.exists() and chroma_db_file.exists()


def rebuild_vector_store_if_needed():
    """
    Rebuilds vector_store/ from knowledge_base/ if vector_store/ is missing.

    This is mainly for Streamlit Cloud deployment.
    We do not push vector_store/ to GitHub because chroma.sqlite3 is too large.
    """
    if vector_store_looks_ready():
        return

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            "vector_store/ is missing and knowledge_base/ was not found. "
            "Upload knowledge_base/ to GitHub or create vector_store/ locally."
        )

    ingest_script = PROJECT_ROOT / "ingest_knowledge_base.py"

    if not ingest_script.exists():
        raise FileNotFoundError(
            "vector_store/ is missing and ingest_knowledge_base.py was not found. "
            "The app cannot rebuild the vector store."
        )

    print("=" * 80)
    print("vector_store/ not found. Rebuilding from knowledge_base/ ...")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(ingest_script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            "Failed to rebuild vector_store/. Check ingest_knowledge_base.py output."
        )

    if not vector_store_looks_ready():
        raise RuntimeError(
            "ingest_knowledge_base.py finished, but vector_store/chroma.sqlite3 was not created."
        )


# =============================================================================
# Chroma collection
# =============================================================================

def get_collection():
    """
    Connects to the local ChromaDB vector store.
    If the vector store is missing, it rebuilds it from knowledge_base/.
    """
    rebuild_vector_store_if_needed()

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load Chroma collection '{COLLECTION_NAME}'. "
            f"Make sure ingest_knowledge_base.py creates this collection name. "
            f"Original error: {e}"
        )

    return collection


# =============================================================================
# Query construction
# =============================================================================

def build_retrieval_query(patient_data: dict) -> str:
    """
    Builds a strong search query from patient/request data.
    """
    query_parts = [
        patient_data.get("drug_name", ""),
        patient_data.get("procedure", ""),
        patient_data.get("requested_service", ""),
        patient_data.get("diagnosis_code", ""),
        patient_data.get("cpt_code", ""),
        patient_data.get("clinical_summary", ""),
    ]

    query = " ".join([part for part in query_parts if part])

    return query.strip()


def infer_body_part_terms(query: str) -> list:
    """
    Infers body part / procedure terms from the query.
    Used for generic source-file boosting instead of only knee-specific boosting.
    """
    query_lower = query.lower()

    candidate_terms = [
        "knee",
        "shoulder",
        "hip",
        "ankle",
        "foot",
        "elbow",
        "wrist",
        "hand",
        "spine",
        "lumbar",
        "cervical",
        "meniscus",
        "meniscal",
        "meniscectomy",
        "rotator cuff",
        "labral",
        "fai",
        "acl",
        "bunion",
        "tfcc",
        "ligament",
        "arthroscopy",
        "arthroscopic",
        "reconstruction",
        "repair",
        "debridement",
        "osteotomy",
    ]

    return [term for term in candidate_terms if term in query_lower]


def score_keyword_boost(document: str, query: str, source_file: str = "") -> int:
    """
    Simple relevance boosting.

    Vector search finds semantically similar chunks, but it can sometimes retrieve
    nearby but wrong policies. This boost helps favor exact procedure/body-part matches.
    """
    doc_lower = document.lower()
    query_lower = query.lower()
    source_lower = source_file.lower()

    score = 0

    important_terms = [
        "medical necessity",
        "conservative",
        "therapy",
        "physical therapy",
        "mri",
        "x-ray",
        "ct",
        "pain",
        "functional",
        "documentation",
        "physical examination",
        "treatment plan",
        "prior authorization",
        "cpt",
        "hcpcs",
    ]

    for term in important_terms:
        if term in query_lower and term in doc_lower:
            score += 2

    inferred_terms = infer_body_part_terms(query)

    for term in inferred_terms:
        if term in doc_lower:
            score += 2

        if term.replace(" ", "-") in source_lower or term in source_lower:
            score += 8

    # Strong boost for documentation guide
    if "medical-records-documentation" in source_lower:
        if any(
            term in query_lower
            for term in ["documentation", "medical necessity", "physical examination", "conservative", "functional"]
        ):
            score += 15

    # Penalize obviously wrong body-part policies
    body_parts = ["knee", "shoulder", "hip", "ankle", "foot", "elbow", "wrist", "hand"]

    query_body_parts = [bp for bp in body_parts if bp in query_lower]

    for bp in body_parts:
        if query_body_parts and bp not in query_body_parts and bp in source_lower:
            score -= 8

    return score


# =============================================================================
# Retrieval
# =============================================================================

def retrieve_policy_context(patient_data: dict, n_results: int = 10, final_k: int = 6):
    """
    Retrieves relevant policy chunks in a balanced way.

    We search from two angles:
    1. Procedure-specific policy search
    2. Documentation/medical necessity requirement search

    Then we intentionally keep both:
    - Procedure policy chunks
    - Documentation requirement chunks
    """
    collection = get_collection()

    procedure_query = build_retrieval_query(patient_data)

    if not procedure_query:
        procedure_query = "prior authorization medical necessity documentation requirements"

    documentation_query = f"""
    medical records documentation requirements medical necessity prior authorization
    symptoms severity of pain functional disability physical examination imaging MRI X-ray CT
    prior conservative therapy treatment plan dates duration failed treatments
    physician treatment plan pre-op discussion diagnostic testing
    {patient_data.get("drug_name", "")}
    {patient_data.get("diagnosis_code", "")}
    """

    procedure_chunks = []
    documentation_chunks = []

    # -------------------------------------------------------------------------
    # 1. Procedure policy retrieval
    # -------------------------------------------------------------------------
    raw_results = collection.query(
        query_texts=[procedure_query],
        n_results=n_results,
    )

    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        source_file = meta.get("source_file", "Unknown file")

        keyword_boost = score_keyword_boost(
            document=doc,
            query=procedure_query,
            source_file=source_file,
        )

        procedure_chunks.append({
            "text": doc,
            "source_file": source_file,
            "source_folder": meta.get("source_folder", "Unknown folder"),
            "chunk_index": meta.get("chunk_index", "Unknown chunk"),
            "distance": distance,
            "keyword_boost": keyword_boost,
            "query_label": "procedure_policy",
        })

    # -------------------------------------------------------------------------
    # 2. Documentation retrieval
    # -------------------------------------------------------------------------
    raw_results = collection.query(
        query_texts=[documentation_query],
        n_results=n_results,
    )

    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        source_file = meta.get("source_file", "Unknown file")

        keyword_boost = score_keyword_boost(
            document=doc,
            query=documentation_query,
            source_file=source_file,
        )

        if "medical-records-documentation" in source_file.lower():
            keyword_boost += 20

        documentation_chunks.append({
            "text": doc,
            "source_file": source_file,
            "source_folder": meta.get("source_folder", "Unknown folder"),
            "chunk_index": meta.get("chunk_index", "Unknown chunk"),
            "distance": distance,
            "keyword_boost": keyword_boost,
            "query_label": "documentation_requirements",
        })

    procedure_chunks = sorted(
        procedure_chunks,
        key=lambda x: (-x["keyword_boost"], x["distance"])
    )

    documentation_chunks = sorted(
        documentation_chunks,
        key=lambda x: (-x["keyword_boost"], x["distance"])
    )

    procedure_target = final_k // 2
    documentation_target = final_k - procedure_target

    selected = procedure_chunks[:procedure_target] + documentation_chunks[:documentation_target]

    seen = set()
    unique_selected = []

    for item in selected:
        unique_key = (item["source_file"], item["chunk_index"])

        if unique_key not in seen:
            seen.add(unique_key)
            unique_selected.append(item)

    return unique_selected[:final_k]


def format_context_for_prompt(retrieved_chunks: list) -> str:
    """
    Converts retrieved policy chunks into clean text for the LLM prompt.
    """
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"""
[POLICY EXCERPT {i}]
Source: {chunk['source_file']}
Folder: {chunk['source_folder']}
Chunk: {chunk['chunk_index']}

{chunk['text']}
"""
        )

    return "\n".join(context_parts)
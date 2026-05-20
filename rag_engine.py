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
    Infers broad service/procedure terms from the query.

    The retrieval boost should not force one specialty's documentation expectations
    onto another specialty. These terms are only used to improve relevance of
    retrieved policy documents, not to score risk.
    """
    query_lower = query.lower()

    candidate_terms = [
        # Orthopedic / musculoskeletal
        "knee", "shoulder", "hip", "ankle", "foot", "elbow", "wrist", "hand",
        "spine", "lumbar", "cervical", "meniscus", "meniscal", "meniscectomy",
        "rotator cuff", "labral", "acl", "bunion", "tfcc", "ligament",
        "arthroscopy", "arthroscopic", "reconstruction", "repair", "debridement",
        "osteotomy",

        # Bariatric / metabolic
        "bariatric", "sleeve gastrectomy", "gastric bypass", "roux-en-y",
        "morbid obesity", "body mass index", "bmi", "weight loss",

        # Gynecology
        "hysterectomy", "myomectomy", "fibroid", "leiomyoma", "abnormal uterine bleeding",
        "endometrial", "pelvic", "uterine", "gynecology",

        # Sleep / respiratory
        "sleep apnea", "cpap", "bipap", "apnea", "respiratory",

        # Oncology / medication / testing
        "oncology", "cancer", "chemotherapy", "radiation", "genetic testing",
        "molecular testing", "infusion", "injection", "drug", "medication",

        # General authorization language
        "medical necessity", "coverage", "benefit", "documentation",
        "prior authorization", "cpt", "hcpcs", "icd-10",
    ]

    return [term for term in candidate_terms if term in query_lower]


def score_keyword_boost(document: str, query: str, source_file: str = "") -> int:
    """
    Simple relevance boosting for retrieval.

    This boost favors coverage/eligibility policies and medical-record documentation
    requirements that match the requested service. It intentionally avoids hardcoding
    universal clinical requirements such as imaging or physical therapy.
    """
    doc_lower = document.lower()
    query_lower = query.lower()
    source_lower = source_file.lower()

    score = 0

    # General policy and documentation terms. These are intentionally broad and
    # do not imply that any one document type is always required.
    important_terms = [
        "medical necessity",
        "coverage",
        "covered",
        "not covered",
        "criteria",
        "indications",
        "limitations",
        "documentation",
        "medical record",
        "clinical notes",
        "prior authorization",
        "authorization",
        "diagnosis",
        "treatment plan",
        "cpt",
        "hcpcs",
        "icd-10",
    ]

    for term in important_terms:
        if term in query_lower and term in doc_lower:
            score += 2

    inferred_terms = infer_body_part_terms(query)

    for term in inferred_terms:
        if term in doc_lower:
            score += 3

        normalized = term.replace(" ", "-")
        if normalized in source_lower or term in source_lower:
            score += 10

    # Strong boost for the general medical-record review guidance.
    if "medical-records-documentation" in source_lower:
        score += 20

    # Prefer omnibus/code/reference documents when CPT/HCPCS is part of the query.
    if any(term in query_lower for term in ["cpt", "hcpcs"]) and any(
        term in source_lower for term in ["omnibus", "codes", "code"]
    ):
        score += 8

    # Soft penalty when the source file appears to be a very different specialty
    # from the query. This only affects retrieval ranking, not model scoring.
    broad_specialties = {
        "orthopedic": ["knee", "shoulder", "hip", "ankle", "meniscus", "arthroscopy", "rotator"],
        "bariatric": ["bariatric", "obesity", "gastrectomy", "gastric"],
        "gynecology": ["hysterectomy", "fibroid", "uterine", "gynecology", "endometrial"],
        "sleep": ["sleep", "apnea", "cpap"],
        "oncology": ["oncology", "cancer", "chemotherapy", "radiation"],
    }

    query_specialties = [
        specialty
        for specialty, terms in broad_specialties.items()
        if any(term in query_lower for term in terms)
    ]

    if query_specialties:
        for specialty, terms in broad_specialties.items():
            if specialty not in query_specialties and any(term in source_lower for term in terms):
                score -= 6

    return score


# =============================================================================
# Retrieval
# =============================================================================

def retrieve_policy_context(patient_data: dict, n_results: int = 10, final_k: int = 6):
    """
    Retrieves relevant policy chunks in a balanced, objective way.

    We search from three angles:
    1. Coverage/eligibility criteria for the requested service
    2. Medical-record documentation requirements used for reviews
    3. Procedure/code-specific policy context

    The retrieval layer should provide context, not pre-decide what is missing.
    """
    collection = get_collection()

    procedure_query = build_retrieval_query(patient_data)

    if not procedure_query:
        procedure_query = "prior authorization coverage criteria medical necessity medical record documentation requirements"

    requested_service = " ".join([
        patient_data.get("drug_name", ""),
        patient_data.get("procedure", ""),
        patient_data.get("requested_service", ""),
        patient_data.get("cpt_code", ""),
    ]).strip()

    diagnosis = patient_data.get("diagnosis_code", "")

    coverage_query = f"""
    coverage criteria medical necessity indications limitations prior authorization eligibility
    requested service procedure CPT HCPCS diagnosis
    {requested_service}
    {diagnosis}
    """

    documentation_query = f"""
    medical records documentation used for reviews required documentation clinical notes
    diagnosis treatment history objective findings test results when applicable
    provider assessment treatment plan dates duration response to treatment
    requested service procedure diagnosis prior authorization
    {requested_service}
    {diagnosis}
    """

    procedure_chunks = []
    coverage_chunks = []
    documentation_chunks = []

    # -------------------------------------------------------------------------
    # 1. Procedure / service retrieval
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
    # 2. Coverage / eligibility retrieval
    # -------------------------------------------------------------------------
    raw_results = collection.query(
        query_texts=[coverage_query],
        n_results=n_results,
    )

    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        source_file = meta.get("source_file", "Unknown file")

        keyword_boost = score_keyword_boost(
            document=doc,
            query=coverage_query,
            source_file=source_file,
        )

        coverage_chunks.append({
            "text": doc,
            "source_file": source_file,
            "source_folder": meta.get("source_folder", "Unknown folder"),
            "chunk_index": meta.get("chunk_index", "Unknown chunk"),
            "distance": distance,
            "keyword_boost": keyword_boost,
            "query_label": "coverage_eligibility",
        })

    # -------------------------------------------------------------------------
    # 3. Medical-record documentation retrieval
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
            keyword_boost += 25

        documentation_chunks.append({
            "text": doc,
            "source_file": source_file,
            "source_folder": meta.get("source_folder", "Unknown folder"),
            "chunk_index": meta.get("chunk_index", "Unknown chunk"),
            "distance": distance,
            "keyword_boost": keyword_boost,
            "query_label": "medical_record_documentation",
        })

    procedure_chunks = sorted(
        procedure_chunks,
        key=lambda x: (-x["keyword_boost"], x["distance"])
    )

    coverage_chunks = sorted(
        coverage_chunks,
        key=lambda x: (-x["keyword_boost"], x["distance"])
    )

    documentation_chunks = sorted(
        documentation_chunks,
        key=lambda x: (-x["keyword_boost"], x["distance"])
    )

    # Keep a balanced mix. For final_k=8: about 3 procedure, 2 coverage, 3 documentation.
    procedure_target = max(1, final_k // 3)
    coverage_target = max(1, final_k // 4)
    documentation_target = final_k - procedure_target - coverage_target

    selected = (
        procedure_chunks[:procedure_target]
        + coverage_chunks[:coverage_target]
        + documentation_chunks[:documentation_target]
    )

    # If deduping leaves fewer than final_k, fill from the combined ranked pool.
    combined_pool = sorted(
        procedure_chunks + coverage_chunks + documentation_chunks,
        key=lambda x: (-x["keyword_boost"], x["distance"])
    )

    seen = set()
    unique_selected = []

    for item in selected + combined_pool:
        unique_key = (item["source_file"], item["chunk_index"])

        if unique_key not in seen:
            seen.add(unique_key)
            unique_selected.append(item)

        if len(unique_selected) >= final_k:
            break

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
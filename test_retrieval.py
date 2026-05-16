from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"


# ---------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

collection = client.get_collection(
    name="uhc_prior_auth_knowledge",
    embedding_function=embedding_function,
)


# ---------------------------------------------------------
# Retrieval function
# ---------------------------------------------------------

def search_knowledge_base(query: str, n_results: int = 5):
    """
    Searches the vector database for the most relevant policy chunks.
    """

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    print("\n" + "=" * 80)
    print("TOP RETRIEVED POLICY CHUNKS")
    print("=" * 80)

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        source_file = meta.get("source_file", "Unknown file")
        source_folder = meta.get("source_folder", "Unknown folder")
        chunk_index = meta.get("chunk_index", "Unknown chunk")

        print(f"\n--- Result {i} ---")
        print(f"Source file   : {source_file}")
        print(f"Source folder : {source_folder}")
        print(f"Chunk index   : {chunk_index}")
        print(f"Distance      : {dist}")
        print("-" * 80)
        print(doc[:1500])
        print("-" * 80)


# ---------------------------------------------------------
# Test queries
# ---------------------------------------------------------

if __name__ == "__main__":
    test_query = """
    Right knee arthroscopy with partial medial meniscectomy CPT 29881.
    Diagnosis M23.221 and M25.561.
    Patient has knee pain, meniscal tear, MRI findings, and conservative therapy.
    What UnitedHealthcare medical necessity documentation is required?
    """

    search_knowledge_base(test_query, n_results=5)
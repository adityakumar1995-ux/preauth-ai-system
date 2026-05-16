from pathlib import Path
import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

POLICY_DIRS = [
    KNOWLEDGE_BASE_DIR / "uhc_policies",
    KNOWLEDGE_BASE_DIR / "uhc_guidelines",
]


# ---------------------------------------------------------
# ChromaDB setup
# ---------------------------------------------------------

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

collection = client.get_or_create_collection(
    name="uhc_prior_auth_knowledge",
    embedding_function=embedding_function,
)


# ---------------------------------------------------------
# PDF reading
# ---------------------------------------------------------

def read_pdf_text(pdf_path: Path) -> str:
    """
    Reads text from a PDF file.

    fitz.open() opens the PDF.
    page.get_text() extracts text from each page.
    """
    text_parts = []

    try:
        doc = fitz.open(pdf_path)

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"\n--- Page {page_number} ---\n{text}")

        doc.close()

    except Exception as e:
        print(f"Could not read {pdf_path.name}: {e}")

    return "\n".join(text_parts)


# ---------------------------------------------------------
# Text chunking
# ---------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """
    Splits long text into smaller overlapping chunks.

    Why?
    LLMs and vector databases work better with smaller chunks.
    Overlap helps avoid losing meaning between chunk boundaries.
    """
    chunks = []

    if not text.strip():
        return chunks

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if start >= text_length:
            break

    return chunks


# ---------------------------------------------------------
# Ingestion
# ---------------------------------------------------------

def ingest_documents():
    """
    Reads all PDFs, chunks them, and stores them in ChromaDB.
    """
    print("=" * 70)
    print("PreAuth.ai — Knowledge Base Ingestion")
    print("=" * 70)

    all_pdfs = []

    for folder in POLICY_DIRS:
        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue

        pdfs = list(folder.glob("*.pdf"))
        print(f"Found {len(pdfs)} PDFs in {folder.name}")
        all_pdfs.extend(pdfs)

    if not all_pdfs:
        print("\nNo PDFs found. Add PDFs to knowledge_base folders and run again.")
        return

    print(f"\nTotal PDFs found: {len(all_pdfs)}")

    total_chunks = 0

    for pdf_index, pdf_path in enumerate(all_pdfs, start=1):
        print(f"\n[{pdf_index}/{len(all_pdfs)}] Reading: {pdf_path.name}")

        text = read_pdf_text(pdf_path)

        if not text.strip():
            print("  No text extracted. Skipping.")
            continue

        chunks = chunk_text(text)

        print(f"  Created {len(chunks)} chunks")

        ids = []
        documents = []
        metadatas = []

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{pdf_path.stem}_{chunk_index}"

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source_file": pdf_path.name,
                "source_folder": pdf_path.parent.name,
                "chunk_index": chunk_index,
            })

        # Add chunks to ChromaDB
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

    print("\n" + "=" * 70)
    print("Ingestion complete")
    print(f"Total PDFs processed: {len(all_pdfs)}")
    print(f"Total chunks stored : {total_chunks}")
    print(f"Vector store path   : {VECTOR_STORE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    ingest_documents()
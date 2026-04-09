import os

# Must be set BEFORE importing HuggingFace/transformers libraries.
# Prevents network calls to HuggingFace Hub — model is already cached locally.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import threading
import logging
from dotenv import load_dotenv
import anthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

logger = logging.getLogger(__name__)

POLICIES_DIR = os.getenv("POLICIES_DIR", "data/pdfs")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION = "policies"

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_embeddings = None
_vectorstore = None
_lock = threading.Lock()


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _lock:
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        with _lock:
            if _vectorstore is None:
                os.makedirs(CHROMA_DIR, exist_ok=True)
                _vectorstore = Chroma(
                    collection_name=COLLECTION,
                    embedding_function=get_embeddings(),
                    persist_directory=CHROMA_DIR,
                )
    return _vectorstore


def get_indexed_filenames() -> list[str]:
    """Return list of PDF filenames already in ChromaDB (no embedding model load needed)."""
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        col = chroma_client.get_collection(COLLECTION)
        docs = col.get(include=["metadatas"])
        sources = set()
        for m in docs["metadatas"]:
            if m and "source" in m:
                sources.add(os.path.basename(m["source"]))
        return sorted(sources)
    except Exception:
        return []


def index_pdfs(pdf_dir: str = None) -> int:
    """Index all PDFs from pdf_dir that are not yet in ChromaDB. Returns chunk count added."""
    pdf_dir = pdf_dir or POLICIES_DIR
    os.makedirs(pdf_dir, exist_ok=True)
    vs = get_vectorstore()

    # Collect already-indexed source paths
    existing = set()
    try:
        docs = vs.get(include=["metadatas"])
        for m in docs["metadatas"]:
            if m and "source" in m:
                existing.add(m["source"])
    except Exception as e:
        logger.warning("Could not read existing vectorstore metadata: %s", e)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    added = 0

    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(pdf_dir, fname)
        if fpath in existing:
            logger.info("Skipping already-indexed: %s", fname)
            continue
        try:
            loader = PyPDFLoader(fpath)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            if chunks:
                vs.add_documents(chunks)
                added += len(chunks)
                logger.info("Indexed %d chunks from %s", len(chunks), fname)
        except Exception as e:
            logger.error("Failed to index %s: %s", fname, e)

    return added


def index_single_pdf(file_path: str) -> int:
    """Index a single PDF file into ChromaDB. Returns chunk count."""
    vs = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    chunks = splitter.split_documents(pages)
    if chunks:
        vs.add_documents(chunks)
        logger.info("Indexed %d chunks from %s", len(chunks), os.path.basename(file_path))
    return len(chunks)


def ensure_pdfs_indexed() -> None:
    """
    Auto-index any PDFs in POLICIES_DIR that are not yet in ChromaDB.
    Called on startup so the RAG agent is ready without manual intervention.
    """
    pdf_dir = POLICIES_DIR
    if not os.path.isdir(pdf_dir):
        logger.warning("Policies directory not found: %s", pdf_dir)
        return

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.info("No PDFs found in %s — skipping auto-index", pdf_dir)
        return

    already_indexed = get_indexed_filenames()
    unindexed = [f for f in pdf_files if f not in already_indexed]

    if not unindexed:
        logger.info("All PDFs already indexed (%d files)", len(already_indexed))
        return

    logger.info("Auto-indexing %d unindexed PDF(s): %s", len(unindexed), unindexed)
    try:
        added = index_pdfs(pdf_dir)
        logger.info("Auto-index complete: %d chunks added", added)
    except Exception as e:
        logger.error("Auto-index failed: %s", e, exc_info=True)


def query_policies(question: str) -> str:
    """Search policy documents and return an LLM-formatted answer with source citation."""
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(question, k=5)

        logger.info("RAG search '%s...' → %d chunks found", question[:60], len(docs))

        if not docs:
            return (
                "⚠️ No policy documents are indexed yet.\n\n"
                "To answer this question, please click **'🔄 Index all PDFs'** in the sidebar. "
                "This loads the policy documents into the knowledge base (~15s on first run)."
            )

        # Build context and collect unique source filenames
        context = "\n\n".join(d.page_content for d in docs)
        sources = sorted({
            os.path.basename(d.metadata.get("source", ""))
            for d in docs
            if d.metadata.get("source")
        })

        prompt = f"""You are a helpful customer support assistant with access to company policy documents.
Answer the following question based only on the provided policy content.
Be specific and cite relevant details from the documents.
If the answer is not clearly stated in the content, say: "This specific detail is not covered in the available policy documents."

Policy content:
{context}

Question: {question}

Answer:"""

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = msg.content[0].text.strip()

        # Append source citation
        if sources:
            source_line = ", ".join(f"*{s}*" for s in sources)
            answer += f"\n\n---\n📄 **Source:** {source_line}"

        return answer

    except anthropic.AuthenticationError:
        logger.error("Anthropic API key is invalid or expired")
        return "⚠️ API key is invalid or expired. Please update ANTHROPIC_API_KEY in your .env file."
    except Exception as e:
        logger.error("RAG agent error for question '%s': %s", question[:80], e, exc_info=True)
        return "Sorry, I couldn't search the policy documents. Please try again."

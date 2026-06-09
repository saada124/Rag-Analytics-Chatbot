import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import pandas as pd
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CACHE_DIR,
    DOCX_DIR,
    PDF_DIR,
    TXT_DIR,
    STRUCTURED_DIR,
)
from models import embed_texts

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================
# CONSTANTS
# =====================================================

COLLECTION_NAME = "sales_knowledge_base"
MANIFEST_PATH = Path(CACHE_DIR) / "ingest_manifest.json"
STRUCTURED_CACHE_DIR = Path(CACHE_DIR) / "structured"
STRUCTURED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        "",
    ],
)

# =====================================================
# CHROMA CLIENT
# =====================================================

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


# =====================================================
# MANIFEST HELPERS
# =====================================================

def load_manifest() -> Dict:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load manifest: %s", exc)
    return {"files": {}}


def save_manifest(manifest: Dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =====================================================
# FILE HASHING
# =====================================================

def sha256_file(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_name(file_path: str) -> str:
    return Path(file_path).stem.lower().replace(" ", "_")


def resolve_path(file_path: str) -> str:
    return str(Path(file_path).resolve())


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text: str) -> str:
    """
    Basic normalization:
    - remove repeated whitespace
    - normalize line breaks
    - remove weird page artifacts
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse long whitespace runs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Final cleanup
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


# =====================================================
# LOADERS
# =====================================================

def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_docx(file_path: str) -> List[Document]:
    loader = Docx2txtLoader(file_path)
    return loader.load()


def load_txt(file_path: str) -> List[Document]:
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_unstructured_file(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    if ext == ".docx":
        return load_docx(file_path)
    if ext == ".txt":
        return load_txt(file_path)

    raise ValueError(f"Unsupported unstructured file type: {ext}")


# =====================================================
# SEMANTIC-STYLE CHUNKING
# =====================================================

def split_documents_semantically(
    documents: List[Document],
) -> List[Document]:
    """
    Structure-aware chunking:
    - keeps page/section boundaries when possible
    - uses overlap
    - preserves metadata
    """
    chunks: List[Document] = []

    for doc in documents:
        source_text = clean_text(doc.page_content)
        if not source_text:
            continue

        base_metadata = dict(doc.metadata or {})
        base_metadata.setdefault("source", base_metadata.get("source", "unknown"))

        split_texts = TEXT_SPLITTER.split_text(source_text)

        for idx, chunk_text in enumerate(split_texts):
            chunk_text = clean_text(chunk_text)
            if not chunk_text:
                continue

            chunk_metadata = dict(base_metadata)
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["chunk_count_hint"] = len(split_texts)

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
            )

    return chunks


# =====================================================
# STRUCTURED DATA HELPERS
# =====================================================

def load_csv_dataframe(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def load_xlsx_dataframe(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)


def load_structured_dataframe(file_path: str) -> pd.DataFrame:
    ext = Path(file_path).suffix.lower()

    if ext == ".csv":
        return load_csv_dataframe(file_path)
    if ext in {".xlsx", ".xls"}:
        return load_xlsx_dataframe(file_path)

    raise ValueError(f"Unsupported structured file type: {ext}")


def save_dataframe_cache(df: pd.DataFrame, file_path: str) -> str:
    """
    Save a dataframe snapshot for analytics use.
    """
    stem = safe_name(file_path)
    cache_path = STRUCTURED_CACHE_DIR / f"{stem}.pkl"
    df.to_pickle(cache_path)
    return str(cache_path)


def dataframe_summary_text(df: pd.DataFrame, file_path: str) -> str:
    """
    Create a compact text summary of a structured file
    so the chatbot can at least discover its schema in retrieval.
    """
    filename = Path(file_path).name
    lines: List[str] = []

    lines.append(f"Structured dataset: {filename}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append("")

    lines.append("Column types:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        lines.append(f"- {col}: {dtype}")

    lines.append("")
    lines.append("First 5 rows:")
    preview = df.head(5).fillna("").to_dict(orient="records")
    for i, row in enumerate(preview, start=1):
        lines.append(f"Row {i}: {row}")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        lines.append("")
        lines.append("Numeric column stats:")
        for col in numeric_cols[:12]:
            series = pd.to_numeric(df[col], errors="coerce")
            lines.append(
                f"- {col}: min={series.min()}, max={series.max()}, mean={series.mean()}"
            )

    return "\n".join(lines)


def build_structured_summary_document(file_path: str, df: pd.DataFrame) -> Document:
    return Document(
        page_content=clean_text(dataframe_summary_text(df, file_path)),
        metadata={
            "source": resolve_path(file_path),
            "file_name": Path(file_path).name,
            "file_type": Path(file_path).suffix.lower().lstrip("."),
            "content_type": "structured_summary",
        },
    )


def delete_existing_by_source(source_path: str) -> None:
    """
    Remove older chunks for the same file before re-ingesting.
    """
    try:
        _collection.delete(where={"source": source_path})
    except Exception:
        # Some chroma versions may be picky about delete filters;
        # failing silently here is better than breaking ingestion.
        pass


def add_documents_to_chroma(documents: List[Document]) -> None:
    if not documents:
        return

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    embeddings = embed_texts(texts)

    ids: List[str] = []
    for i, meta in enumerate(metadatas):
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", i)
        fingerprint = hashlib.sha256(
            f"{source}:{chunk_index}:{texts[i][:200]}".encode("utf-8")
        ).hexdigest()[:24]
        ids.append(fingerprint)

    _collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

def ingest_unstructured_file(
    file_path: str,
    force: bool = False,
) -> int:
    """
    Ingest PDF/DOCX/TXT into Chroma.
    Returns number of chunks added.
    """
    file_path = resolve_path(file_path)
    if not os.path.isfile(file_path):
        logger.warning("Skipping missing file: %s", file_path)
        return 0

    ext = Path(file_path).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        return 0

    manifest = load_manifest()
    file_hash = sha256_file(file_path)
    previous = manifest["files"].get(file_path)

    if previous and previous.get("sha256") == file_hash and not force:
        logger.info("Already indexed: %s", file_path)
        return 0

    if force or previous:
        delete_existing_by_source(file_path)

    raw_docs = load_unstructured_file(file_path)

    cleaned_docs: List[Document] = []
    for doc in raw_docs:
        text = clean_text(doc.page_content)
        if not text:
            continue

        metadata = dict(doc.metadata or {})
        metadata["source"] = file_path
        metadata["file_name"] = Path(file_path).name
        metadata["file_type"] = ext.lstrip(".")
        metadata["content_type"] = "unstructured"
        metadata["ingested_at"] = now_iso()
        metadata["sha256"] = file_hash

        cleaned_docs.append(Document(page_content=text, metadata=metadata))

    chunked_docs = split_documents_semantically(cleaned_docs)
    add_documents_to_chroma(chunked_docs)

    manifest["files"][file_path] = {
        "sha256": file_hash,
        "file_type": ext.lstrip("."),
        "kind": "unstructured",
        "chunks": len(chunked_docs),
        "last_ingested_at": now_iso(),
    }
    save_manifest(manifest)

    logger.info("Ingested unstructured file: %s (%d chunks)", file_path, len(chunked_docs))
    return len(chunked_docs)


def ingest_structured_file(
    file_path: str,
    force: bool = False,
) -> int:
    """
    Ingest CSV/XLSX as:
    - cached DataFrame for analytics
    - one summary document into Chroma for discovery
    Returns number of summary docs added (0 or 1).
    """
    file_path = resolve_path(file_path)
    if not os.path.isfile(file_path):
        logger.warning("Skipping missing file: %s", file_path)
        return 0

    ext = Path(file_path).suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        return 0

    manifest = load_manifest()
    file_hash = sha256_file(file_path)
    previous = manifest["files"].get(file_path)

    if previous and previous.get("sha256") == file_hash and not force:
        logger.info("Already indexed structured file: %s", file_path)
        return 0

    if force or previous:
        delete_existing_by_source(file_path)

    df = load_structured_dataframe(file_path)
    cache_path = save_dataframe_cache(df, file_path)

    summary_doc = build_structured_summary_document(file_path, df)
    summary_doc.metadata.update(
        {
            "sha256": file_hash,
            "cached_dataframe_path": cache_path,
            "ingested_at": now_iso(),
        }
    )

    add_documents_to_chroma([summary_doc])

    manifest["files"][file_path] = {
        "sha256": file_hash,
        "file_type": ext.lstrip("."),
        "kind": "structured",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "cached_dataframe_path": cache_path,
        "last_ingested_at": now_iso(),
    }
    save_manifest(manifest)

    logger.info("Ingested structured file: %s", file_path)
    return 1


# =====================================================
# DIRECTORY INGESTION
# =====================================================

def collect_files_from_directory(directory: str, extensions: Tuple[str, ...]) -> List[str]:
    directory_path = Path(directory)
    if not directory_path.exists():
        return []

    collected: List[str] = []
    for root, _, files in os.walk(directory):
        for name in files:
            if Path(name).suffix.lower() in extensions:
                collected.append(str(Path(root) / name))
    return collected


def ingest_all(
    force: bool = False,
    include_documents: bool = True,
    include_structured: bool = True,
) -> Dict[str, int]:
    """
    Ingest everything under the configured folders.
    """
    results = {
        "unstructured_chunks": 0,
        "structured_docs": 0,
    }

    if include_documents:
        document_files = []
        document_files.extend(collect_files_from_directory(PDF_DIR, (".pdf",)))
        document_files.extend(collect_files_from_directory(DOCX_DIR, (".docx",)))
        document_files.extend(collect_files_from_directory(TXT_DIR, (".txt",)))

        for file_path in document_files:
            results["unstructured_chunks"] += ingest_unstructured_file(
                file_path=file_path,
                force=force,
            )

    if include_structured:
        structured_files = collect_files_from_directory(
            STRUCTURED_DIR,
            (".csv", ".xlsx", ".xls"),
        )

        for file_path in structured_files:
            results["structured_docs"] += ingest_structured_file(
                file_path=file_path,
                force=force,
            )

    return results


def get_collection_count() -> int:
    try:
        return _collection.count()
    except Exception:
        return 0


def clear_collection() -> None:
    """
    Use carefully. This wipes the full knowledge base.
    """
    try:
        all_ids = _collection.get(include=[])["ids"]
        if all_ids:
            _collection.delete(ids=all_ids)
    except Exception as exc:
        logger.warning("Failed to clear collection: %s", exc)


if __name__ == "__main__":
    stats = ingest_all(force=False)
    print("\nIngestion complete.")
    print(stats)
    print(f"Chroma count: {get_collection_count()}")
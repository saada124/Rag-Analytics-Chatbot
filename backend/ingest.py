import json
import hashlib
from pathlib import Path

import pandas as pd

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PDF_DIR,
    DOCX_DIR,
    TXT_DIR,
    STRUCTURED_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CACHE_DIR,
    MIN_CHUNK_CHARS,
)

from models import embedding_function

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding_function,
        )
    return _vectorstore


def load_pdf(file_path):
    return PyPDFLoader(str(file_path)).load()


def load_docx(file_path):
    return Docx2txtLoader(str(file_path)).load()


def load_txt(file_path):
    try:
        return TextLoader(str(file_path), encoding="utf-8").load()
    except (UnicodeDecodeError, RuntimeError):
        return TextLoader(str(file_path), encoding="latin-1").load()


def load_document(file_path):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    if suffix == ".docx":
        return load_docx(file_path)
    if suffix == ".txt":
        return load_txt(file_path)
    return []

#CHUNK DOCUMENTS
def chunk_documents(documents):
    chunks = text_splitter.split_documents(documents)
    cleaned = []
    for chunk in chunks:
        content = (chunk.page_content or "").strip()
        if len(content) < MIN_CHUNK_CHARS:
            continue
        chunk.page_content = content
        src = chunk.metadata.get("source", "")
        if src:
            chunk.metadata["source"] = Path(src).name
        cleaned.append(chunk)
    return cleaned


def _deterministic_ids(chunks):
    ids = []
    seen = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = seen.get(source, 0)
        seen[source] = idx + 1
        digest = hashlib.sha1(
            f"{source}:{idx}:{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        ids.append(digest)
    return ids

def ingest_documents():
    vectorstore = get_vectorstore()
    documents = []

    for folder in [PDF_DIR, DOCX_DIR, TXT_DIR]:
        for file_path in Path(folder).glob("*"):
            if not file_path.is_file():
                continue
            print(f"[INFO] Loading: {file_path.name}")
            try:
                docs = load_document(file_path)
            except Exception as e:
                # One unreadable / corrupt file must not abort the whole run.
                print(f"[ERROR] Skipping {file_path.name}: {e}")
                continue
            documents.extend(docs)

    if not documents:
        print("[INFO] No documents found.")
        return

    chunks = chunk_documents(documents)
    if not chunks:
        print("[INFO] No non-empty chunks to index.")
        return

    ids = _deterministic_ids(chunks)
    print(f"[INFO] Generated {len(chunks)} chunks")
    # Passing stable ids makes re-ingestion idempotent (upsert, no duplicates).
    vectorstore.add_documents(chunks, ids=ids)
    print("[INFO] Documents indexed.")


def dataframe_summary(df):
    summary = {"rows": int(len(df)), "columns": list(df.columns)}
    preview = df.head(5).where(pd.notna(df.head(5)), "").to_dict(orient="records")
    return {"summary": summary, "preview": preview}

def _detect_separator(file_path):

    candidates = (";", ",", "\t", "|")
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for raw in f:
                if raw.strip():
                    lines.append(raw)
                if len(lines) >= 20:
                    break
    except OSError:
        return ","
    if not lines:
        return ","

    best_sep, best_key = ",", (-1, -1.0)
    for sep in candidates:
        counts = [len(line.split(sep)) for line in lines]
        header_fields = counts[0]
        if header_fields < 2:
            continue
        consistency = sum(1 for c in counts if c == header_fields) / len(counts)
        # Prefer a delimiter that is consistent first, then yields more columns.
        key = (1 if consistency >= 0.6 else 0, header_fields * consistency)
        if key > best_key:
            best_key, best_sep = key, sep
    return best_sep


def load_csv(file_path):
    sep = _detect_separator(file_path)

    read_kwargs = dict(
        engine="python",          # tolerant parser; handles odd quoting/lines
        na_values=["NULL", "null", ""],
        on_bad_lines="warn",      # warn (not silently skip) so data loss is visible
    )

    def _read(separator):
        try:
            return pd.read_csv(file_path, encoding="utf-8", sep=separator, **read_kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="latin-1", sep=separator, **read_kwargs)

    df = _read(sep)
    if df.shape[1] == 1:
        only_col = str(df.columns[0])
        for alt in (";", "\t", "|", ","):
            if alt != sep and alt in only_col:
                alt_df = _read(alt)
                if alt_df.shape[1] > df.shape[1]:
                    df, sep = alt_df, alt
                    break

    print(f"[INFO] {Path(file_path).name}: delimiter={sep!r} -> {df.shape[1]} columns")

    # Clean trailing commas from column names.
    df.columns = df.columns.astype(str).str.rstrip(",").str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.rstrip(",").strip() if isinstance(x, str) else x)
        df[col] = df[col].replace({"NULL": pd.NA, "null": pd.NA, "": pd.NA})
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna().sum()
        if non_null == 0:
            continue
        if converted.notna().sum() == non_null:
            df[col] = converted
            continue
        # European number format fallback: thousands '.' + decimal ',' e.g.
        # "1.234,56" or "1234,56". Only applied when the WHOLE column parses
        # cleanly this way, so US-style decimals are never corrupted.
        euro = pd.to_numeric(
            df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if euro.notna().sum() == non_null:
            df[col] = euro

    return df


def load_excel(file_path):
    return pd.read_excel(file_path)


def ingest_structured_files():
    vectorstore = get_vectorstore()
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    ids = []

    for file_path in Path(STRUCTURED_DIR).glob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".csv":
                df = load_csv(file_path)
            elif suffix in [".xlsx", ".xls"]:
                df = load_excel(file_path)
            else:
                continue

            cache_file = cache_dir / f"{file_path.stem}.pkl"
            df.to_pickle(cache_file)

            summary = dataframe_summary(df)
            text = json.dumps(summary, indent=2, ensure_ascii=False, default=str)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "type": "structured",
                        "cache": str(cache_file),
                    },
                )
            )
            # Stable id per structured source for idempotent re-ingestion.
            ids.append(
                hashlib.sha1(f"structured:{file_path.name}".encode("utf-8")).hexdigest()
            )
            print(f"[INFO] Loaded {file_path.name}")

        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")

    if documents:
        vectorstore.add_documents(documents, ids=ids)
        print("[INFO] Structured datasets indexed.")


def ingest_all():
    print("\n[INFO] Starting ingestion\n")
    ingest_documents()
    ingest_structured_files()
    print("\n[INFO] Ingestion complete.\n")


def get_collection_count():
    vectorstore = get_vectorstore()
    try:
        return vectorstore._collection.count()
    except Exception:
        # Public-API fallback if the private attribute changes.
        return len(vectorstore.get().get("ids", []))

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":
    ingest_all()
    print()
    print("Total Chunks:")
    print(get_collection_count())

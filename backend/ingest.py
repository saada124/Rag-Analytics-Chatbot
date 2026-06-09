import json
from pathlib import Path

import pandas as pd

from langchain_core.documents import Document

from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from config import (
    PDF_DIR,
    DOCX_DIR,
    TXT_DIR,
    STRUCTURED_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from models import embedding_function

# ==========================================================
# TEXT SPLITTER
# ==========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

# ==========================================================
# VECTOR STORE
# ==========================================================

_vectorstore = None

def get_vectorstore():

    global _vectorstore

    if _vectorstore is None:

        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding_function
        )

    return _vectorstore

# ==========================================================
# PDF
# ==========================================================

def load_pdf(file_path):

    loader = PyPDFLoader(str(file_path))

    return loader.load()

# ==========================================================
# DOCX
# ==========================================================

def load_docx(file_path):

    loader = Docx2txtLoader(
        str(file_path)
    )

    return loader.load()

# ==========================================================
# TXT
# ==========================================================

def load_txt(file_path):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    return loader.load()

# ==========================================================
# DOCUMENT LOADER
# ==========================================================

def load_document(file_path):

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)

    if suffix == ".docx":
        return load_docx(file_path)

    if suffix == ".txt":
        return load_txt(file_path)

    return []

# ==========================================================
# CHUNK DOCUMENTS
# ==========================================================

def chunk_documents(documents):

    return text_splitter.split_documents(
        documents
    )

# ==========================================================
# PDF/DOCX/TXT INGESTION
# ==========================================================

def ingest_documents():

    vectorstore = get_vectorstore()

    documents = []

    folders = [
        PDF_DIR,
        DOCX_DIR,
        TXT_DIR
    ]

    for folder in folders:

        for file_path in Path(folder).glob("*"):

            print(
                f"[INFO] Loading: "
                f"{file_path.name}"
            )

            docs = load_document(
                file_path
            )

            documents.extend(
                docs
            )

    if not documents:

        print(
            "[INFO] No documents found."
        )

        return

    chunks = chunk_documents(
        documents
    )

    print(
        f"[INFO] Generated "
        f"{len(chunks)} chunks"
    )

    vectorstore.add_documents(
        chunks
    )

    print(
        "[INFO] Documents indexed."
    )

# ==========================================================
# DATAFRAME SUMMARY
# ==========================================================

def dataframe_summary(df):

    summary = {
        "rows": len(df),
        "columns": list(df.columns)
    }

    preview = (
        df.head(5)
        .fillna("")
        .to_dict(
            orient="records"
        )
    )

    return {
        "summary": summary,
        "preview": preview
    }

# ==========================================================
# CSV
# ==========================================================

def load_csv(file_path):

    return pd.read_csv(
        file_path
    )

def load_excel(file_path):

    return pd.read_excel(
        file_path
    )

# ==========================================================
# STRUCTURED INGESTION
# ==========================================================

def ingest_structured_files():

    vectorstore = get_vectorstore()

    cache_dir = (
        Path("cache")
        / "dataframes"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = []

    for file_path in Path(
        STRUCTURED_DIR
    ).glob("*"):

        suffix = (
            file_path
            .suffix
            .lower()
        )

        try:

            if suffix == ".csv":

                df = load_csv(
                    file_path
                )

            elif suffix in [
                ".xlsx",
                ".xls"
            ]:

                df = load_excel(
                    file_path
                )

            else:

                continue

            cache_file = (
                cache_dir
                / f"{file_path.stem}.pkl"
            )

            df.to_pickle(
                cache_file
            )

            summary = dataframe_summary(
                df
            )

            text = json.dumps(
                summary,
                indent=2,
                ensure_ascii=False
            )

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source":
                        file_path.name,
                        "type":
                        "structured",
                        "cache":
                        str(cache_file)
                    }
                )
            )

            print(
                f"[INFO] Loaded "
                f"{file_path.name}"
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{file_path.name}: "
                f"{e}"
            )

    if documents:

        vectorstore.add_documents(
            documents
        )

        print(
            "[INFO] Structured "
            "datasets indexed."
        )

# ==========================================================
# INGEST ALL
# ==========================================================

def ingest_all():

    print(
        "\n[INFO] Starting ingestion\n"
    )

    ingest_documents()

    ingest_structured_files()

    print(
        "\n[INFO] Ingestion complete.\n"
    )

# ==========================================================
# STATS
# ==========================================================

def get_collection_count():

    vectorstore = get_vectorstore()

    return vectorstore._collection.count()

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    ingest_all()

    print()

    print(
        "Total Chunks:"
    )

    print(
        get_collection_count()
    )
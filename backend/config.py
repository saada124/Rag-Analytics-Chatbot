import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "openai/gpt-4o-mini"
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "./chroma_db"
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", 1000)
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", 200)
)

TOP_K = int(
    os.getenv("TOP_K", 20)
)

TOP_N = int(
    os.getenv("TOP_N", 5)
)

EMBEDDING_MODEL_NAME = (
    "intfloat/multilingual-e5-large"
)

RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-v2-m3"
)

PDF_DIR = "data/documents/pdf"
DOCX_DIR = "data/documents/docx"
TXT_DIR = "data/documents/txt"

STRUCTURED_DIR = "data/structured"

CACHE_DIR = "cache"
LOG_DIR = "logs"

DIRECTORIES = [
    CHROMA_PATH,
    PDF_DIR,
    DOCX_DIR,
    TXT_DIR,
    STRUCTURED_DIR,
    CACHE_DIR,
    LOG_DIR,
]

for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)
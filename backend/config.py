# config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App Details
APP_NAME = "RAG & Analytics Chatbot"
APP_VERSION = "2.0.0"
HOST = "0.0.0.0"
PORT = 8000

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Chunking Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Retrieval Parameters
VECTOR_SEARCH_TOP_K = int(os.getenv("TOP_K", 20))
RERANKER_TOP_K = int(os.getenv("TOP_N", 5))

# V2: Multi-query Expansion
MULTI_QUERY_COUNT = 4
MULTI_QUERY_TOP_K = 20

# Model Selection
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Memory Configuration
MEMORY_SIZE = 10

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
CSV_DIR = os.path.join(BASE_DIR, "data", "csv")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
CHROMA_COLLECTION = "documents"

# Specific formats for document types
PDF_DIR = os.path.join(DOCUMENTS_DIR, "pdf")
DOCX_DIR = os.path.join(DOCUMENTS_DIR, "docx")
TXT_DIR = os.path.join(DOCUMENTS_DIR, "txt")
STRUCTURED_DIR = CSV_DIR
CHROMA_DIR = CHROMA_PATH

# Analytics Fallback
FALLBACK_ROW_THRESHOLD = 0
FALLBACK_ENABLED = True

# Ensure data directories exist on startup
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DOCX_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
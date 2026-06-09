from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    APP_NAME,
    APP_VERSION,
    HOST,
    PORT
)

from ingest import (
    ingest_all,
    get_collection_count
)

from router import (
    route_query
)

from models import (
    memory
)

from analytics import load_dataframes

# ==========================================================
# LIFESPAN CONTEXT MANAGER
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Pre-loading and normalizing DataFrames...")
    load_dataframes()
    yield
    print("[SHUTDOWN] Stopping application...")

# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan
)

# ==========================================================
# REQUEST MODELS
# ==========================================================

class ChatRequest(BaseModel):
    message: str


# ==========================================================
# HEALTH
# ==========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_documents": get_collection_count()
    }


# ==========================================================
# INGEST
# ==========================================================

@app.post("/ingest")
def ingest():
    ingest_all()
    # Refresh dataframes after ingestion
    load_dataframes()
    return {
        "status": "success",
        "vector_documents": get_collection_count()
    }


# ==========================================================
# CHAT
# ==========================================================

@app.post("/chat")
def chat(request: ChatRequest):
    result = route_query(request.message)
    return result


# ==========================================================
# MEMORY
# ==========================================================

@app.post("/clear-memory")
def clear_memory():
    memory.clear()
    return {
        "status": "memory cleared"
    }


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def root():
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "running"
    }


# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=True
    )
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Pre-loading and normalizing DataFrames...")
    load_dataframes()
    yield
    print("[SHUTDOWN] Stopping application...")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_documents": get_collection_count()
    }


@app.post("/ingest")
def ingest():
    ingest_all()
    #refresh dataframes after ingestion
    load_dataframes()
    return {
        "status": "success",
        "vector_documents": get_collection_count()
    }

@app.post("/chat")
def chat(request: ChatRequest):
    result = route_query(request.message)
    return result

@app.post("/clear-memory")
def clear_memory():
    memory.clear()
    return {
        "status": "memory cleared"
    }

@app.get("/")
def root():
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "running"
    }


#test
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=True
    )
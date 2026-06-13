import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
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

from session import (
    session_manager
)

from analytics import load_dataframes

SESSION_COOKIE = "session_id"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30


def _get_session_id(request: Request) -> tuple[str, bool]:
    """Return (session_id, is_new). Reuses the cookie if present, else mints one."""
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        return sid, False
    return uuid.uuid4().hex, True


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] Pre-loading and normalizing DataFrames...")
    load_dataframes()
    yield
    logger.info("[SHUTDOWN] Stopping application...")

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
def chat(request: ChatRequest, http_request: Request, response: Response):
    session_id, is_new = _get_session_id(http_request)
    result = route_query(request.message, session_id=session_id)
    _set_session_cookie(response, session_id)
    return result

@app.post("/clear-memory")
def clear_memory(http_request: Request):
    session_id, _ = _get_session_id(http_request)
    # Clear only this session's history, not everyone's.
    session_manager.get_memory(session_id).clear()
    return {
        "status": "memory cleared"
    }

@app.get("/")
def root():
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "yekhdem mrigel"
    }


#test
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=True,
        reload_includes=["*.py"],
        reload_excludes=[
            "chroma_db/*", "chroma_db/**",
            "cache/*", "cache/**",
            "data/*", "data/**",
            "__pycache__/*", "**/__pycache__/**",
            "*.pkl", "*.sqlite3", "*.sqlite3-journal", "*.bin", "*.log",
        ],
    )

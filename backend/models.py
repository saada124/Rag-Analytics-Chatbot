import logging

logger = logging.getLogger(__name__)

import threading
import torch
import numpy as np
from typing import Sequence, List
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.callbacks import Callbacks

from config import (
    OPENROUTER_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

logger.info(f"[INFO] Running on {DEVICE}")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found.")

llm_client = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model=MODEL_NAME,
    temperature=0,
    max_tokens=4000,
)


class E5Embeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" if not t.startswith("passage: ") else t for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {text}" if not text.startswith("query: ") else text
        return super().embed_query(prefixed)


logger.info(f"[INFO] Loading embeddings: {EMBEDDING_MODEL_NAME}")

embedding_function = E5Embeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

logger.info("[INFO] Embeddings ready.")


class CrossEncoderReranker(BaseDocumentCompressor):
    model_name: str
    device: str
    top_n: int = 5
    _model: object = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load model once at init time
        self._model = CrossEncoder(self.model_name, device=self.device)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]


logger.info(f"[INFO] Loading reranker: {RERANKER_MODEL_NAME}")
reranker_compressor = CrossEncoderReranker(model_name=RERANKER_MODEL_NAME, device=DEVICE)
logger.info("[INFO] Reranker ready.")

#SEMANTIC CACHE
class SemanticCache:
    def __init__(self, embedding_func, threshold: float = 0.98, max_size: int = 1000):
        self.embedding_func = embedding_func
        self.threshold = threshold
        self.max_size = max_size
        self._lock = threading.Lock()
        self._responses = []          # parallel to rows of self._matrix
        self._matrix = None           # np.ndarray of shape (n, dim), or None

    def get_cached_response(self, query: str):
        with self._lock:
            if self._matrix is None or len(self._responses) == 0:
                return None
            matrix = self._matrix
            responses = self._responses

        query_emb = np.asarray(self.embedding_func.embed_query(query), dtype=np.float32)
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_emb)
        denom[denom == 0] = 1e-12
        similarities = (matrix @ query_emb) / denom

        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] >= self.threshold:
            logger.info(f"[CACHE HIT] Similarity: {similarities[best_idx]:.4f}")
            return responses[best_idx]
        return None

    def add_to_cache(self, query: str, response: dict):
        query_emb = np.asarray(self.embedding_func.embed_query(query), dtype=np.float32)
        with self._lock:
            self._responses.append(response)
            if self._matrix is None:
                self._matrix = query_emb.reshape(1, -1)
            else:
                self._matrix = np.vstack([self._matrix, query_emb])

            # Evict oldest entries (FIFO) once we exceed the cap.
            if len(self._responses) > self.max_size:
                overflow = len(self._responses) - self.max_size
                self._responses = self._responses[overflow:]
                self._matrix = self._matrix[overflow:]

    def clear(self):
        with self._lock:
            self._responses = []
            self._matrix = None


semantic_cache = SemanticCache(embedding_func=embedding_function)

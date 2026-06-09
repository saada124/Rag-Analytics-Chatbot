import torch
from collections import deque
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
    MEMORY_SIZE
)

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"[INFO] Running on {DEVICE}")

# ==========================================================
# LANGCHAIN CHAT LLM (OpenRouter / OpenAI wrapper)
# ==========================================================

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found.")

llm_client = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model=MODEL_NAME,
    temperature=0
)

# ==========================================================
# CUSTOM EMBEDDINGS (Handles E5 prefixes natively)
# ==========================================================

class E5Embeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" if not t.startswith("passage: ") else t for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {text}" if not text.startswith("query: ") else text
        return super().embed_query(prefixed)

print(f"[INFO] Loading embeddings: {EMBEDDING_MODEL_NAME}")

embedding_function = E5Embeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

print("[INFO] Embeddings ready.")

# ==========================================================
# CUSTOM DOCUMENT COMPRESSOR (CrossEncoder Reranker)
# ==========================================================

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
        callbacks: Callbacks = None
    ) -> Sequence[Document]:
        if not documents:
            return []
        
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_n]]

print(f"[INFO] Loading reranker: {RERANKER_MODEL_NAME}")
reranker_compressor = CrossEncoderReranker(model_name=RERANKER_MODEL_NAME, device=DEVICE)
print("[INFO] Reranker ready.")

# ==========================================================
# CONVERSATION MEMORY
# ==========================================================

class ConversationMemory:
    def __init__(self, size=10):
        self.messages = deque(maxlen=size)

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def get_history(self) -> str:
        if not self.messages:
            return ""
        lines = []
        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def clear(self):
        self.messages.clear()

memory = ConversationMemory(MEMORY_SIZE)
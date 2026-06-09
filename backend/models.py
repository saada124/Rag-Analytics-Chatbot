import torch

from openai import OpenAI

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from config import (
    OPENROUTER_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Using device: {DEVICE}")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from .env"
    )

client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

print(
    f"[INFO] Loading embedding model: "
    f"{EMBEDDING_MODEL_NAME}"
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=DEVICE
)

print("[INFO] Embedding model loaded.")

print(
    f"[INFO] Loading reranker model: "
    f"{RERANKER_MODEL_NAME}"
)

reranker_model = CrossEncoder(
    RERANKER_MODEL_NAME,
    device=DEVICE
)

print("[INFO] Reranker model loaded.")

def embed_texts(texts):
    """
    Generate embeddings for a list of texts.

    Returns:
        List[List[float]]
    """

    if not texts:
        return []

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()

def rewrite_query(query: str) -> str:
    """
    Rewrite user query for retrieval.
    """

    prompt = f"""
You are a retrieval optimization assistant.

Rewrite the user's question into a concise
search-friendly query.

Do not answer the question.

User Question:
{query}

Search Query:
"""

    response = client_llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite questions for semantic search."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    rewritten = (
        response.choices[0]
        .message.content
        .strip()
    )

    return rewritten

def generate_answer(
    question: str,
    context: str
) -> str:
    """
    Generate grounded answer.
    """

    prompt = f"""
You are a sales company assistant.

Answer ONLY using the provided context.

If the answer is not contained
in the context, say:

'I could not find this information
in the available documents.'

Context:
{context}

Question:
{question}
"""

    response = client_llm.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only supplied context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response.choices[0]
        .message.content
        .strip()
    )

# RERANK DOCUMENTS
def rerank_documents(
    query: str,
    documents: list,
    top_n: int = 5
):

    if not documents:
        return []

    pairs = [
        [query, doc]
        for doc in documents
    ]

    scores = reranker_model.predict(
        pairs,
        show_progress_bar=False
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score in ranked[:top_n]
    ]

if __name__ == "__main__":

    query = "What is the warranty policy?"
    rewritten = rewrite_query(query)
    print()
    print("Original:")
    print(query)
    print()
    print("Rewritten:")
    print(rewritten)
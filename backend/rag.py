from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import ContextualCompressionRetriever

from config import (
    VECTOR_SEARCH_TOP_K,
    RERANKER_TOP_K,
    MAX_CONTEXT_CHARS,
)

from ingest import get_vectorstore
from session import ConversationMemory

from models import (
    llm_client,
    reranker_compressor,
)

NO_ANSWER = (
    "Je suis désolé, mais la documentation ne contient pas les informations "
    "nécessaires pour répondre à cette question."
)

reranker_compressor.top_n = RERANKER_TOP_K

vectorstore = get_vectorstore()
base_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K})

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker_compressor,
    base_retriever=base_retriever,
)

# History-Aware Query Condensation Chain
condense_prompt = ChatPromptTemplate.from_messages([
    ("system", """\
Given the conversation history below and a follow-up question from the user, \
rewrite the follow-up question into a fully standalone, keyword-rich search query \
that can be understood without any prior context.

Rules:
- If the follow-up question contains pronouns (it, this, that, they, the new version, the issue...), \
replace them with the actual referenced entities from the history.
- Preserve all domain-specific terms: product names, dates, countries, technical terms.
- Do NOT answer the question. Return ONLY the rewritten search query string, nothing else."""),
    ("user", "Conversation History:\n{history}\n\nFollow-up Question:\n{query}"),
])
query_condenser = condense_prompt | llm_client | StrOutputParser()

# QA Answering Chain — Strict Grounding + Source Citation + French
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """\
You are a sales company assistant. You must follow these rules strictly:

1. Answer ONLY using the information present in the provided context below.
2. If the answer cannot be found in the context, you MUST respond with exactly: \
"Je suis désolé, mais la documentation ne contient pas les informations nécessaires pour répondre à cette question."
3. Do NOT use your pre-trained knowledge to fill gaps in the context. Never invent facts, numbers, or policies.
4. For every factual claim you make, add an inline citation of the source document in brackets, e.g. [chroma_knowledge_base.pdf].
5. Your ENTIRE response must be written in French.
6. FORMAT: be concise - avoid long intro/outro paragraphs. When the user asks several things, address each requested item under its own short bold title, followed by detailed bullet points containing the concrete facts."""),
    ("user", "Conversation History:\n{history}\n\nContext:\n{context}\n\nQuestion:\n{question}"),
])
qa_chain = qa_prompt | llm_client | StrOutputParser()


def retrieve_documents(query: str) -> List:
    return compression_retriever.invoke(query)


def _build_context(reranked_docs) -> str:
    """Concatenate chunks up to a character budget so we never overflow the
    model context window on large chunks."""
    parts = []
    total = 0
    for doc in reranked_docs:
        source = doc.metadata.get("source", "unknown")
        piece = f"[{source}]\n{doc.page_content}"
        if total + len(piece) > MAX_CONTEXT_CHARS and parts:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def retrieve_context(query: str, history: str = ""):
    if history.strip():
        condensed_query = query_condenser.invoke(
            {"query": query, "history": history}
        ).strip()
    else:
        condensed_query = query.strip()

    print(f"[RAG] Condensed query: {condensed_query}")

    reranked_docs = retrieve_documents(condensed_query)
    context = _build_context(reranked_docs)

    return {
        "query": query,
        "condensed_query": condensed_query,
        "documents": reranked_docs,
        "context": context,
    }


def extract_sources(documents: List) -> List[str]:
    sources = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("source")
        if source and source not in sources:
            sources.append(source)
    return sources


def answer_rag_question(question: str, memory: ConversationMemory):
    history = memory.get_history()

    retrieval = retrieve_context(question, history=history)

    if not retrieval["context"].strip():
        answer = NO_ANSWER
    else:
        answer = qa_chain.invoke(
            {
                "question": question,
                "context": retrieval["context"],
                "history": history,
            }
        ).strip()

    memory.add_user(question)
    memory.add_assistant(answer)

    return {
        "answer": answer,
        "sources": extract_sources(retrieval["documents"]),
        "condensed_query": retrieval["condensed_query"],
    }

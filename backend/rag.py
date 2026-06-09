from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import ContextualCompressionRetriever

from config import (
    VECTOR_SEARCH_TOP_K,
    RERANKER_TOP_K
)

from ingest import get_vectorstore

from models import (
    llm_client,
    reranker_compressor,
    memory
)

# ==========================================================
# RETRIEVERS
# ==========================================================

# Configure custom top_n on our reranker compressor
reranker_compressor.top_n = RERANKER_TOP_K

# Get base retriever from LangChain Chroma
vectorstore = get_vectorstore()
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": VECTOR_SEARCH_TOP_K}
)

# Build contextual compression retriever with LangChain
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker_compressor,
    base_retriever=base_retriever
)

# ==========================================================
# LCEL CHAINS
# ==========================================================

# History-Aware Query Condensation Chain
# Resolves pronouns and ambiguous references from conversation history
# into a fully standalone, keyword-rich search query.
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
    ("user", "Conversation History:\n{history}\n\nFollow-up Question:\n{query}")
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
5. Your ENTIRE response must be written in French."""),
    ("user", "Conversation History:\n{history}\n\nContext:\n{context}\n\nQuestion:\n{question}")
])
qa_chain = qa_prompt | llm_client | StrOutputParser()

# ==========================================================
# MAIN INTERFACES
# ==========================================================

def retrieve_documents(query: str) -> List:
    return compression_retriever.invoke(query)

def retrieve_context(query: str, history: str = ""):
    # Condense the query using conversation history to resolve pronouns/references
    if history.strip():
        condensed_query = query_condenser.invoke({"query": query, "history": history}).strip()
    else:
        condensed_query = query.strip()

    print(f"[RAG] Condensed query: {condensed_query}")

    # Retrieve and rerank docs using the condensed query
    reranked_docs = retrieve_documents(condensed_query)

    # Format context with source metadata appended to each chunk
    context_parts = []
    for doc in reranked_docs:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[{source}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    return {
        "query": query,
        "condensed_query": condensed_query,
        "documents": reranked_docs,
        "context": context
    }

def extract_sources(documents: List) -> List[str]:
    sources = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("source")
        if source and source not in sources:
            sources.append(source)
    return sources

def answer_rag_question(question: str):
    history = memory.get_history()

    # Pass history into retrieve_context so the condenser can resolve references
    retrieval = retrieve_context(question, history=history)

    # Invoke LCEL Answer Generation
    answer = qa_chain.invoke({
        "question": question,
        "context": retrieval["context"],
        "history": history
    }).strip()

    # Update Memory
    memory.add_user(question)
    memory.add_assistant(answer)

    return {
        "answer": answer,
        "sources": extract_sources(retrieval["documents"]),
        "condensed_query": retrieval["condensed_query"]
    }
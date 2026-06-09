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

# Query Rewriter Chain
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You optimize search queries for semantic retrieval. Return only the rewritten query, nothing else."),
    ("user", "Rewrite this query keeping products, dates, customer names, and locations:\n\n{query}")
])
query_rewriter = rewrite_prompt | llm_client | StrOutputParser()

# QA Answering Chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sales company assistant. Answer ONLY using the context. If the answer is not found, say so clearly. Your response must ALWAYS be in French."),
    ("user", "Conversation History:\n{history}\n\nContext:\n{context}\n\nQuestion:\n{question}")
])
qa_chain = qa_prompt | llm_client | StrOutputParser()

# ==========================================================
# MAIN INTERFACES
# ==========================================================

def retrieve_documents(query: str) -> List:
    return compression_retriever.invoke(query)

def retrieve_context(query: str):
    # Rewrite the query
    rewritten_query = query_rewriter.invoke({"query": query}).strip()

    # Retrieve docs
    reranked_docs = retrieve_documents(rewritten_query)

    # Format context
    context_parts = [doc.page_content for doc in reranked_docs]
    context = "\n\n".join(context_parts)

    return {
        "query": query,
        "rewritten_query": rewritten_query,
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
    retrieval = retrieve_context(question)
    history = memory.get_history()

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
        "rewritten_query": retrieval["rewritten_query"]
    }
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from analytics import analyze_query
from rag import (
    retrieve_context,
    answer_rag_question
)

from models import (
    memory,
    llm_client,
    semantic_cache
)

class RouterClassification(BaseModel):
    route: Literal["RAG", "ANALYTICS", "HYBRID"] = Field(
        ...,
        description="The target route. RAG is for policy/documentation questions; ANALYTICS is for calculations, counts, sums, averages, lists of products/customers; HYBRID is for questions asking for both text descriptions/policies AND data stats."
    )

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for a business intelligence chatbot.
Classify the user's message into one of:
- RAG: policy, warranty, contracts, company procedures, text contents.
- ANALYTICS: totals, averages, sales, lists of top products/customers, calculations, AND specific record lookups (e.g. searching by ID, email, name, or text fields).
- HYBRID: questions requiring both data/numbers AND document/policy context."""),
    ("user", "{query}")
])

router_chain = router_prompt | llm_client.with_structured_output(RouterClassification)

def classify_query(query: str) -> str:
    try:
        result = router_chain.invoke({"query": query})
        return result.route
    except Exception as e:
        print(f"[ROUTER ERROR] {e}")
        return "RAG"

#hybrid response
hybrid_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sales company assistant. Answer the user question by combining BOTH sources (Analytics and Documents). Your response must ALWAYS be in French."),
    ("user", "Analytics Data:\n{analytics_context}\n\nDocuments Context:\n{rag_context}\n\nConversation History:\n{history}\n\nQuestion:\n{question}")
])

hybrid_chain = hybrid_prompt | llm_client | StrOutputParser()

def handle_analytics(query: str) -> dict:
    result = analyze_query(query)
    answer = result.get("answer", "No answer found.")

    memory.add_user(query)
    memory.add_assistant(answer)

    return {
        "type": "analytics",
        "answer": answer,
        "data": result.get("data")
    }

def handle_rag(query: str) -> dict:
    result = answer_rag_question(query)
    return {
        "type": "rag",
        "answer": result["answer"],
        "sources": result["sources"]
    }

def handle_hybrid(query: str) -> dict:
    analytics_result = analyze_query(query)
    history = memory.get_history()
    retrieval_result = retrieve_context(query, history=history)

    analytics_context = analytics_result.get("answer", "")
    rag_context = retrieval_result.get("context", "")

    answer = hybrid_chain.invoke({
        "question": query,
        "analytics_context": analytics_context,
        "rag_context": rag_context,
        "history": history
    }).strip()

    memory.add_user(query)
    memory.add_assistant(answer)

    return {
        "type": "hybrid",
        "answer": answer,
        "analytics": analytics_result,
        "sources": [
            doc.metadata.get("source")
            for doc in retrieval_result["documents"]
            if doc.metadata.get("source")
        ]
    }

def route_query(query: str) -> dict:
    #check semantic cache first
    cached = semantic_cache.get_cached_response(query)
    if cached is not None:
        return cached

    query_type = classify_query(query)
    print(f"[ROUTER] {query_type}")

    if query_type == "ANALYTICS":
        response = handle_analytics(query)
    elif query_type == "HYBRID":
        response = handle_hybrid(query)
    else:
        response = handle_rag(query)

    #store in semantic cache
    semantic_cache.add_to_cache(query, response)
    return response
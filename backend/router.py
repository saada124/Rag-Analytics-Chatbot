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
    llm_client
)

# ==========================================================
# PYDANTIC ROUTER MODEL
# ==========================================================

class RouterClassification(BaseModel):
    route: Literal["RAG", "ANALYTICS", "HYBRID"] = Field(
        ...,
        description="The target route. RAG is for policy/documentation questions; ANALYTICS is for calculations, counts, sums, averages, lists of products/customers; HYBRID is for questions asking for both text descriptions/policies AND data stats."
    )

# ==========================================================
# QUERY CLASSIFIER CHAIN
# ==========================================================

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for a business intelligence chatbot.
Classify the user's message into one of:
- RAG: policy, warranty, contracts, company procedures, text contents.
- ANALYTICS: totals, averages, sales, lists of top products/customers, calculations.
- HYBRID: questions requiring both data/numbers AND document/policy context (e.g. 'top-selling products and their warranty terms')."""),
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

# ==========================================================
# HYBRID RESPONSE CHAIN
# ==========================================================

hybrid_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sales company assistant. Answer the user question by combining BOTH sources (Analytics and Documents)."),
    ("user", "Analytics Data:\n{analytics_context}\n\nDocuments Context:\n{rag_context}\n\nConversation History:\n{history}\n\nQuestion:\n{question}")
])

hybrid_chain = hybrid_prompt | llm_client | StrOutputParser()

# ==========================================================
# ROUTE HANDLERS
# ==========================================================

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
    retrieval_result = retrieve_context(query)
    history = memory.get_history()

    analytics_context = analytics_result.get("answer", "")
    rag_context = retrieval_result.get("context", "")

    # Invoke LCEL hybrid answer generation
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

# ==========================================================
# MAIN ROUTER
# ==========================================================

def route_query(query: str) -> dict:
    query_type = classify_query(query)
    print(f"[ROUTER] {query_type}")

    if query_type == "ANALYTICS":
        return handle_analytics(query)
    if query_type == "HYBRID":
        return handle_hybrid(query)
    
    return handle_rag(query)
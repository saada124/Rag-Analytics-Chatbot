from typing import Literal
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from analytics import analyze_query
from rag import retrieve_context, answer_rag_question
from session import session_manager, ConversationMemory

from models import (
    llm_client,
    semantic_cache,
)


class RouterClassification(BaseModel):
    route: Literal["RAG", "ANALYTICS", "HYBRID"] = Field(
        ...,
        description=(
            "The target route. RAG is for policy/documentation questions; "
            "ANALYTICS is for calculations, counts, sums, averages, lists of "
            "products/customers; HYBRID is for questions asking for both text "
            "descriptions/policies AND data stats."
        ),
    )


router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for a business intelligence chatbot.
Classify the user's message into one of:
- RAG: policy, warranty, contracts, company procedures, text contents.
- ANALYTICS: totals, averages, sales, lists of top products/customers, calculations, AND specific record lookups (e.g. searching by ID, email, name, or text fields).
- HYBRID: questions requiring both data/numbers AND document/policy context."""),
    ("user", "{query}"),
])

router_chain = router_prompt | llm_client.with_structured_output(RouterClassification)


def classify_query(query: str) -> str:
    try:
        result = router_chain.invoke({"query": query})
        return result.route
    except Exception as e:
        print(f"[ROUTER ERROR] {e}")
        return "RAG"


hybrid_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a sales company assistant. Answer the user's question by combining BOTH sources (Analytics and Documents). Your response must ALWAYS be in French.

FORMAT:
- Be concise. Do NOT write long introductory or concluding paragraphs.
- When the user asks several things, address EACH requested item under its own short bold title, immediately followed by one or more detailed bullet points.
- Put the concrete figures (totals, names, amounts, percentages) directly inside the bullets.

GROUNDING (very important):
- Use ONLY the Analytics Data for any number, total, ranking, name or amount. NEVER invent, estimate, or label a value as 'hypothetical'.
- If a requested figure is missing from the Analytics Data, say briefly that it is not available in the data - do not fabricate it."""),
    ("user", "Analytics Data:\n{analytics_context}\n\nDocuments Context:\n{rag_context}\n\nConversation History:\n{history}\n\nQuestion:\n{question}"),
])

hybrid_chain = hybrid_prompt | llm_client | StrOutputParser()


def handle_analytics(query: str, memory: ConversationMemory) -> dict:
    result = analyze_query(query)
    answer = result.get("answer", "No answer found.")

    memory.add_user(query)
    memory.add_assistant(answer)

    return {"type": "analytics", "answer": answer, "data": result.get("data")}


def handle_rag(query: str, memory: ConversationMemory) -> dict:
    # answer_rag_question updates the session memory internally.
    result = answer_rag_question(query, memory)
    return {"type": "rag", "answer": result["answer"], "sources": result["sources"]}


def handle_hybrid(query: str, memory: ConversationMemory) -> dict:
    history = memory.get_history()

    # Run analytics and RAG retrieval in parallel.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_analytics = executor.submit(analyze_query, query)
        future_rag = executor.submit(retrieve_context, query, history)
        analytics_result = future_analytics.result()
        retrieval_result = future_rag.result()

    analytics_context = analytics_result.get("answer", "")
    rag_context = retrieval_result.get("context", "")

    answer = hybrid_chain.invoke({
        "question": query,
        "analytics_context": analytics_context,
        "rag_context": rag_context,
        "history": history,
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
        ],
    }


def route_query(query: str, session_id: str = None) -> dict:
    """Route a query for a specific session.

    `session_id` isolates conversation memory per user/conversation. Pass a
    stable id (e.g. user id or chat id) from your web layer.
    """
    memory = session_manager.get_memory(session_id)
    query_type = classify_query(query)
    print(f"[ROUTER] {query_type}")

    if query_type == "ANALYTICS":
        # Do NOT cache analytics: results are data-dependent and time-sensitive,
        # and a semantically-similar query may need a different answer.
        return handle_analytics(query, memory)

    if query_type == "HYBRID":
        # Hybrid includes analytics data, so it is not cached either.
        return handle_hybrid(query, memory)

    # RAG route: safe to cache (shared documentation knowledge).
    cached = semantic_cache.get_cached_response(query)
    if cached is not None:
        # Still record the turn so history-aware condensation stays correct.
        memory.add_user(query)
        memory.add_assistant(cached.get("answer", ""))
        return cached

    response = handle_rag(query, memory)
    semantic_cache.add_to_cache(query, response)
    return response

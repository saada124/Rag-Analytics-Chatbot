# RAG Chatbot — Full Build Plan

---

## Overview

A hybrid chatbot that answers two fundamentally different question types: knowledge questions (answered by searching documents) and analytics questions (answered by running calculations on structured data). A query router classifies every incoming message and dispatches it to the right engine. GPT-4o mini powers the LLM layer throughout.

The system ships in three versions:

- **V1** — complete working pipeline: document ingestion, CSV ingestion, vector search, reranking, analytics engine, query routing, hybrid retrieval, and basic conversation memory.
- **V1.1** — single targeted improvement on top of V1: a fallback classifier that catches failed analytics queries and retries them through the RAG pipeline instead of returning empty results.
- **V2** — three quality upgrades built after V1.1 is stable: multi-query retrieval expansion, improved natural-language filter extraction, and conversation summary memory replacing the raw message window.

Each version is independently testable. Build V1 fully, verify it end to end, then add V1.1, then V2.

---

## Technology stack

### Backend framework
- **FastAPI** — async HTTP server, handles the chat endpoint and file ingestion endpoints
- **Python 3.11+**
- **Uvicorn** — ASGI server to run FastAPI

### LLM
- **GPT-4o mini** via the OpenAI Python SDK (`openai>=1.0.0`)
- Used for: query routing, filter extraction, query rewriting, answer synthesis, and (in V2) query expansion and memory summarization
- Always pass `response_format={"type": "json_object"}` when expecting structured JSON output — the router, filter extractor, and query expander all need this

### Embeddings
- **intfloat/multilingual-e5-large** via `sentence-transformers`
- Supports English, French, and Arabic — the three languages likely in Tunisian business documents
- Runs fully locally, no API call needed
- Important: this model requires a prefix on input text. Prepend `"query: "` to query strings and `"passage: "` to document chunks before encoding. Skipping this silently degrades retrieval quality.
- If your corpus is English-only, swap to **BAAI/bge-base-en-v1.5** for faster inference — same prefix convention applies

### Vector database
- **ChromaDB** (`chromadb>=0.4`) — local persistent store
- Stores chunks with their embeddings and metadata (source filename, chunk index, page number)
- Persists to disk automatically at `chroma_db/`

### Reranker
- **BAAI/bge-reranker-base** via `sentence-transformers` CrossEncoder
- Takes (query, chunk) pairs and re-scores them — significantly more accurate than cosine similarity alone
- Runs locally, no API call needed
- Load once at startup; calling `.predict()` on a batch is fast enough for real-time use

### Document processing
- **LangChain** (`langchain`, `langchain-community`) — document loaders and text splitters
- **pypdf** — PDF parsing
- **python-docx** — DOCX parsing
- **unstructured** — optional, for complex PDFs with mixed layouts, embedded tables, or scanned pages

### Structured data processing
- **Pandas** — DataFrames for all CSV/XLSX operations
- **NumPy** — supporting numerical calculations
- **openpyxl** — required by Pandas to read `.xlsx` files

### API and utilities
- `openai` — GPT-4o mini calls
- `python-dotenv` — loads `.env` file into environment
- `pydantic` — request and response validation in FastAPI
- `fastapi`, `uvicorn[standard]`

---

## Folder structure

```
project/
│
├── app.py                        # FastAPI app, route definitions, main query handler
├── config.py                     # All constants in one place
├── requirements.txt
├── .env                          # OPENAI_API_KEY — never commit this
│
├── data/
│   ├── documents/                # Drop PDF, DOCX, TXT files here for ingestion
│   └── csv/                      # Drop CSV, XLSX files here for ingestion
│
├── ingest/
│   ├── document_ingest.py        # Loads documents, chunks them, embeds, writes to ChromaDB
│   └── csv_ingest.py             # Loads CSVs/XLSX into Pandas DataFrames, registers them
│
├── retrieval/
│   ├── vector_store.py           # ChromaDB wrapper: add_chunks(), search()
│   ├── reranker.py               # bge-reranker-base wrapper: rerank(query, chunks) -> top_k
│   └── query_rewriter.py         # V1: rewrite(). V2: also expand() for multi-query
│
├── analytics/
│   ├── dataframe_manager.py      # Registry of named DataFrames, load_all() on startup
│   ├── filters.py                # extract_filters(query) -> FilterObject, apply_filters(df, filters)
│   └── calculations.py           # sum_col(), avg_col(), group_by(), growth_pct(), profit_pct()
│
├── routing/
│   └── query_router.py           # classify(query) -> "rag" | "analytics" | "hybrid"
│
├── memory/
│   └── conversation_store.py     # V1: last_n_messages(). V2: also update_summary(), get_summary()
│
├── llm/
│   └── client.py                 # Thin wrapper around openai.chat.completions.create()
│
└── chroma_db/                    # ChromaDB persists here automatically
```

---

## config.py — all tunable constants in one place

```python
# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
VECTOR_SEARCH_TOP_K = 20        # candidates fetched from ChromaDB before reranking
RERANKER_TOP_K = 5              # chunks passed to LLM after reranking

# V2: multi-query expansion
MULTI_QUERY_COUNT = 4           # number of sub-queries to generate per user question
MULTI_QUERY_TOP_K = 20          # top_k per sub-query before dedup and reranking

# Models
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "gpt-4o-mini"

# Memory
MEMORY_WINDOW = 10              # V1: number of raw messages to retain per session
SUMMARY_MAX_TOKENS = 300        # V2: max tokens for the running conversation summary

# Paths
DOCUMENTS_DIR = "data/documents"
CSV_DIR = "data/csv"
CHROMA_PATH = "chroma_db"
CHROMA_COLLECTION = "documents"

# Analytics fallback (V1.1)
FALLBACK_ROW_THRESHOLD = 0      # if Pandas returns <= this many rows, trigger RAG fallback
FALLBACK_ENABLED = True         # set to False to disable fallback during debugging
```

---

## V1 — Core pipeline

### Step 1 — Document ingestion (`ingest/document_ingest.py`)

Load every file from `data/documents/`, split into chunks, embed them, and write to ChromaDB.

**File loading.** Use LangChain loaders dispatched by file extension: `PyPDFLoader` for `.pdf`, `Docx2txtLoader` for `.docx`, `TextLoader` for `.txt`. Loop over every file in `DOCUMENTS_DIR` and route to the correct loader. Collect all loaded documents into a flat list before chunking.

**Chunking.** Use `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)`. This splitter tries to break on paragraph boundaries, then sentences, then words, falling back to characters only when necessary. The overlap ensures that a sentence split across two chunks still appears in full in at least one of them. For the first version this is sufficient. An upgrade to `SemanticChunker` is planned after V1 is stable.

**Embedding.** Load the embedding model once at module import time — do not re-instantiate it per request. Call `model.encode(texts, normalize_embeddings=True)` in batches (batch size 32 for GPU, 8 for CPU-only). Always normalize — ChromaDB uses cosine similarity, which requires unit vectors. Remember the e5 prefix: prepend `"passage: "` to every chunk text before encoding.

**Writing to ChromaDB.** Get or create the collection by name. Generate a unique ID for each chunk: `f"{filename}_chunk_{i}"`. Call `collection.add(embeddings=..., documents=..., metadatas=..., ids=...)`. Metadata should include `{"source": filename, "chunk_index": i, "page": page_number_or_0}`.

**Deduplication.** If the same file is ingested twice, ChromaDB's `add()` will raise on duplicate IDs. Use `collection.upsert()` instead of `collection.add()` so re-ingesting a file updates existing chunks rather than erroring.

Run ingestion either at startup or via the `/ingest/documents` POST endpoint.

### Step 2 — CSV/XLSX ingestion (`ingest/csv_ingest.py`, `analytics/dataframe_manager.py`)

Load every file from `data/csv/` into a Pandas DataFrame. Register each one by its filename stem (no extension) in a global dict: `{"sales_2026": df, "products": df, ...}`. This dict is the registry that the analytics engine reads at query time.

**Normalizing on load.** Apply these transformations to every DataFrame immediately after loading:

```python
# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Normalize all string values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip().str.lower()

# Parse date columns automatically
for col in df.columns:
    if any(kw in col for kw in ["date", "time", "period", "month", "year"]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
```

Normalizing at ingestion time means filter extraction never needs to worry about casing, leading spaces, or column name variations. This single step eliminates the most common class of analytics failures.

Call `dataframe_manager.load_all()` once at FastAPI startup via a `lifespan` handler.

### Step 3 — ChromaDB and reranker wrappers

**`retrieval/vector_store.py`**

```python
def search(query: str, top_k: int = VECTOR_SEARCH_TOP_K) -> list[dict]:
    # Prepend e5 query prefix before encoding
    prefixed = f"query: {query}"
    query_embedding = embed_model.encode([prefixed], normalize_embeddings=True)[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count())   # guard against small collections
    )
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "id": results["ids"][0][i],
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    return chunks
```

**`retrieval/reranker.py`**

```python
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder(RERANKER_MODEL)

def rerank(query: str, chunks: list[dict], top_k: int = RERANKER_TOP_K) -> list[dict]:
    if not chunks:
        return []
    pairs = [(query, c["text"]) for c in chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]
```

### Step 4 — Query rewriter (`retrieval/query_rewriter.py`)

Rewrites the raw user question into a clean, specific search query before it hits ChromaDB. This step costs roughly 50 tokens but measurably improves recall on short and ambiguous questions.

```python
def rewrite(query: str) -> str:
    system = (
        "You are a search query optimizer. Rewrite the user's question as a clean, "
        "specific search query that would retrieve relevant document chunks. "
        "Remove conversational filler. Be specific. Return only the rewritten query, nothing else."
    )
    return llm_client.chat(system=system, messages=[{"role": "user", "content": query}], max_tokens=100)
```

### Step 5 — Analytics filter extractor (`analytics/filters.py`)

This is the most critical and fragile piece of the analytics pipeline. Its job is to turn a natural-language sentence into a structured JSON object that Pandas can execute without ambiguity.

**Filter schema.** Every field must have a default value so the object is always complete:

```json
{
  "dataframe": "sales_2026",
  "filters": {
    "product": null,
    "country": null,
    "date_start": null,
    "date_end": null,
    "exclude_status": [],
    "group_by": null
  },
  "calculation": "sum",
  "target_column": "revenue"
}
```

**Extraction call.** Use `response_format={"type": "json_object"}`. The system prompt must include the list of available DataFrames and their column names, injected at call time from `dataframe_manager`. Example:

```
You are a data filter extractor for a business analytics chatbot.
Available DataFrames and their columns:
- sales_2026: order_id, date, product, country, revenue, quantity, status
- products: product_id, name, category, warranty_months, price

Extract a filter object from the user's query using this schema exactly:
{"dataframe": string, "filters": {"product": null, "country": null, "date_start": "YYYY-MM-DD or null",
"date_end": "YYYY-MM-DD or null", "exclude_status": [], "group_by": null},
"calculation": "sum|mean|count|group_by", "target_column": string}

All fields are required. Use null for fields not mentioned. Return valid JSON only.

Examples:
User: "Total sales in Tunisia during March 2026"
{"dataframe": "sales_2026", "filters": {"product": null, "country": "tunisia", "date_start": "2026-03-01",
"date_end": "2026-03-31", "exclude_status": [], "group_by": null}, "calculation": "sum", "target_column": "revenue"}

User: "Average order value in France excluding cancelled orders"
{"dataframe": "sales_2026", "filters": {"product": null, "country": "france", "date_start": null,
"date_end": null, "exclude_status": ["cancelled"], "group_by": null}, "calculation": "mean", "target_column": "revenue"}

User: "Top countries by sales"
{"dataframe": "sales_2026", "filters": {"product": null, "country": null, "date_start": null,
"date_end": null, "exclude_status": [], "group_by": "country"}, "calculation": "group_by", "target_column": "revenue"}
```

**Applying filters (`apply_filters`).** Each filter is applied only if the field is non-null:

```python
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if filters.get("product"):
        df = df[df["product"] == filters["product"].lower()]
    if filters.get("country"):
        df = df[df["country"] == filters["country"].lower()]
    if filters.get("date_start"):
        try:
            df = df[df["date"] >= pd.to_datetime(filters["date_start"])]
        except Exception:
            pass   # log and skip, don't crash
    if filters.get("date_end"):
        try:
            df = df[df["date"] <= pd.to_datetime(filters["date_end"])]
        except Exception:
            pass
    if filters.get("exclude_status"):
        exclusions = [s.lower() for s in filters["exclude_status"]]
        df = df[~df["status"].isin(exclusions)]
    return df
```

**Calculations (`calculations.py`).** After filtering, run the requested calculation:

```python
def run_calculation(df: pd.DataFrame, calculation: str, target_column: str, group_by: str = None) -> str:
    if df.empty:
        return None   # signals the fallback classifier to take over

    if calculation == "sum":
        result = df[target_column].sum()
        return f"Total {target_column}: {result:,.2f}"

    if calculation == "mean":
        result = df[target_column].mean()
        return f"Average {target_column}: {result:,.2f}"

    if calculation == "count":
        return f"Count: {len(df)}"

    if calculation == "group_by" and group_by:
        result = df.groupby(group_by)[target_column].sum().sort_values(ascending=False)
        return result.to_string()

    return None
```

Returning `None` from `run_calculation` is the signal to the fallback classifier. An empty DataFrame or an unrecognized calculation both return `None`.

### Step 6 — Query router (`routing/query_router.py`)

Classifies every query into one of three routes. For analytics and hybrid queries, it also extracts the filter object in the same call to avoid a round trip.

```python
def classify(query: str) -> dict:
    system = """
You are a query classifier for a business intelligence chatbot.
Classify the user's message as one of:
- "rag": questions about policies, product descriptions, procedures, text content
- "analytics": questions asking for numbers, totals, averages, rankings, trends
- "hybrid": questions requiring both a number AND explanatory text
  (e.g. "top-selling laptops and their warranty terms")

For "analytics" and "hybrid", also extract the filter object.
Return JSON: {"route": "rag|analytics|hybrid", "filters": {...} or null}
Return valid JSON only. No explanation.
"""
    response = llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": query}],
        response_format={"type": "json_object"},
        max_tokens=300
    )
    result = json.loads(response)
    # Validate and fill defaults on filters if present
    if result.get("filters"):
        result["filters"] = fill_filter_defaults(result["filters"])
    return result
```

**`fill_filter_defaults`** ensures every key exists before the filters reach Pandas:

```python
def fill_filter_defaults(filters: dict) -> dict:
    defaults = {
        "dataframe": None,
        "filters": {
            "product": None, "country": None,
            "date_start": None, "date_end": None,
            "exclude_status": [], "group_by": None
        },
        "calculation": "sum",
        "target_column": "revenue"
    }
    for key, val in defaults.items():
        if key not in filters:
            filters[key] = val
    if "filters" in filters:
        for key, val in defaults["filters"].items():
            if key not in filters["filters"]:
                filters["filters"][key] = val
    return filters
```

### Step 7 — Memory (`memory/conversation_store.py`)

V1 keeps the last N raw messages per session in memory, keyed by session ID.

```python
sessions: dict[str, list[dict]] = {}

def add_message(session_id: str, role: str, content: str):
    sessions.setdefault(session_id, [])
    sessions[session_id].append({"role": role, "content": content})

def get_context(session_id: str, n: int = MEMORY_WINDOW) -> list[dict]:
    return sessions.get(session_id, [])[-n:]

def clear(session_id: str):
    sessions.pop(session_id, None)
```

Inject the result of `get_context()` into every LLM call as the messages history immediately before the current user message.

Note: this dict is in-memory only. On server restart all sessions are lost. For production, replace the dict with Redis or SQLite — the function signatures stay identical, only the storage backend changes.

### Step 8 — LLM client (`llm/client.py`)

A thin wrapper so no other module imports `openai` directly. Centralizing this makes it easy to swap the model, add retry logic, or add token counting later.

```python
from openai import OpenAI
from config import LLM_MODEL

_client = OpenAI()  # reads OPENAI_API_KEY from environment automatically

def chat(
    system: str,
    messages: list[dict],
    response_format: dict = None,
    max_tokens: int = 1000
) -> str:
    kwargs = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": max_tokens
    }
    if response_format:
        kwargs["response_format"] = response_format
    response = _client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
```

### Step 9 — Main query handler (`app.py`)

The complete flow for a single `/chat` POST request:

1. Receive `{"session_id": "...", "message": "..."}`.
2. Load memory context: `context = memory.get_context(session_id)`.
3. Classify: `routing = query_router.classify(message)` → get route and filters.
4. Branch by route:

**RAG path:**
```
rewritten_query = query_rewriter.rewrite(message)
candidates = vector_store.search(rewritten_query, top_k=VECTOR_SEARCH_TOP_K)
top_chunks = reranker.rerank(rewritten_query, candidates, top_k=RERANKER_TOP_K)
answer = llm_client.chat(system=RAG_SYSTEM_PROMPT, messages=context + [user_msg_with_chunks])
```

**Analytics path:**
```
filters = routing["filters"]
df = dataframe_manager.get(filters["dataframe"])
filtered_df = analytics.apply_filters(df, filters["filters"])
result_str = analytics.run_calculation(filtered_df, filters["calculation"], filters["target_column"], filters["filters"].get("group_by"))
if result_str is None:
    # fallback handled in V1.1
    answer = "I couldn't find data matching those criteria."
else:
    answer = llm_client.chat(system=ANALYTICS_SYSTEM_PROMPT, messages=context + [user_msg_with_result])
```

**Hybrid path:**
```python
# Run both paths concurrently
import asyncio

async def run_hybrid(message, filters, context):
    rag_task = asyncio.create_task(run_rag_path(message))
    analytics_task = asyncio.create_task(run_analytics_path(message, filters))
    rag_chunks, analytics_result = await asyncio.gather(rag_task, analytics_task)
    # Merge into one prompt
    combined_prompt = f"Data result:\n{analytics_result}\n\nRelevant context:\n{format_chunks(rag_chunks)}"
    return llm_client.chat(system=HYBRID_SYSTEM_PROMPT, messages=context + [{"role": "user", "content": combined_prompt + "\n\nUser question: " + message}])
```

5. Store both turns: `memory.add_message(session_id, "user", message)` and `memory.add_message(session_id, "assistant", answer)`.
6. Return `{"answer": answer, "route": routing["route"], "sources": [c["metadata"]["source"] for c in top_chunks]}`.

---

## V1 — FastAPI endpoints

```
POST /chat
Body:     {"session_id": str, "message": str}
Response: {"answer": str, "route": str, "sources": list[str]}

POST /ingest/documents
Body:     multipart file upload (PDF, DOCX, or TXT)
Response: {"status": "ok", "chunks_added": int}

POST /ingest/csv
Body:     multipart file upload (CSV or XLSX)
Response: {"status": "ok", "dataframe": str, "rows": int, "columns": list[str]}

GET /health
Response: {"status": "ok", "chroma_docs": int, "dataframes": list[str]}
```

The `/health` endpoint verifies that ingestion worked and that all expected DataFrames are loaded. Check it after every server restart before testing queries.

---

## V1 — Build order

Follow this exactly. Each step is independently testable before moving on.

1. Create folder structure, `config.py`, `.env` with `OPENAI_API_KEY`. Verify the environment loads.
2. Build `llm/client.py`. Test with a single hardcoded call that returns "hello" from GPT-4o mini.
3. Build `ingest/document_ingest.py`. Drop a PDF in `data/documents/`, run ingestion, query ChromaDB directly to verify chunks are stored.
4. Build `ingest/csv_ingest.py` and `dataframe_manager.py`. Load a CSV, print the head and dtypes, verify column normalization worked.
5. Build `retrieval/vector_store.py`. Run a test query, print the top 5 results with their distances.
6. Build `retrieval/reranker.py`. Feed the results from step 5 into the reranker, print the reordered list, verify the order changed.
7. Build `retrieval/query_rewriter.py`. Print the rewritten version of 5–10 representative queries.
8. Build `analytics/filters.py`. Run 10 test queries through `extract_filters()`, print the JSON, manually verify each one. Iterate on the prompt until extraction is reliable.
9. Build `routing/query_router.py`. Run 15 queries covering all three routes. Check that "rag"/"analytics"/"hybrid" is assigned correctly for each.
10. Build `memory/conversation_store.py` (V1 version).
11. Wire everything together in `app.py`. Test `/chat` with one query of each type.
12. Manually test 20–30 realistic queries. Fix any routing or filter extraction errors before shipping V1.

---

## V1.1 — Fallback classifier

V1.1 is a single targeted addition to the analytics path in `app.py`. No new files are needed. The change is roughly 15 lines.

### What it solves

When the analytics engine runs filters and returns zero rows, V1 responds with an empty or meaningless answer. This happens more often than expected: the user mentions a product name that is spelled differently in the DataFrame, uses a country name that doesn't match the stored values (e.g. "Tunisia" vs "TN"), uses a relative date the extractor misparses, or asks about a metric that doesn't exist in the available DataFrames.

V1.1 catches this condition and reruns the query through the RAG pipeline as a fallback. The user gets a useful answer from documents instead of silence.

### Implementation

In `app.py`, modify the analytics branch of the main query handler:

```python
# Analytics path with fallback (V1.1)
filters = routing["filters"]
fallback_triggered = False
result_str = None

try:
    df_name = filters.get("dataframe")
    df = dataframe_manager.get(df_name)

    if df is None:
        raise ValueError(f"DataFrame '{df_name}' not found in registry")

    filtered_df = analytics.apply_filters(df, filters["filters"])
    result_str = analytics.run_calculation(
        filtered_df,
        filters["calculation"],
        filters["target_column"],
        filters["filters"].get("group_by")
    )
except Exception as e:
    logger.warning(f"Analytics execution failed: {e}. Triggering RAG fallback.")
    result_str = None

if result_str is None and FALLBACK_ENABLED:
    # Log the failed filters for later prompt improvement
    logger.info(f"FALLBACK triggered | original: {message} | filters: {filters}")
    fallback_triggered = True

    # Rerun as RAG
    rewritten_query = query_rewriter.rewrite(message)
    candidates = vector_store.search(rewritten_query, top_k=VECTOR_SEARCH_TOP_K)
    top_chunks = reranker.rerank(rewritten_query, candidates, top_k=RERANKER_TOP_K)
    answer = llm_client.chat(
        system=RAG_SYSTEM_PROMPT,
        messages=context + [build_rag_user_message(message, top_chunks)]
    )
elif result_str is None:
    answer = "I couldn't find data matching those criteria. Try rephrasing or check the available data."
else:
    answer = llm_client.chat(
        system=ANALYTICS_SYSTEM_PROMPT,
        messages=context + [build_analytics_user_message(message, result_str)]
    )
```

Include `"fallback": fallback_triggered` in the API response so the frontend can optionally signal to the user that the answer came from documents rather than data.

### Using the fallback logs

Every fallback log entry contains the original query and the extracted filters that produced zero rows. After a week of use, read through these logs — most failures will cluster around the same patterns:

- A product name that GPT-4o mini writes as "laptop" but your column stores as "laptop pro 15"
- Country names stored as codes ("TN", "FR") when the model extracts full names ("Tunisia", "France")
- Relative date expressions ("last month") that aren't resolved to absolute dates

Fix each pattern either by improving the extraction prompt with more examples, or by normalizing the stored values at ingestion time. After two or three rounds of this, the fallback rate should drop to near zero for normal use.

### V1.1 — Build checklist

1. Add the `FALLBACK_ENABLED` and `FALLBACK_ROW_THRESHOLD` constants to `config.py` if not already present.
2. Add the fallback logic to the analytics path in `app.py` as shown above.
3. Add structured logging for fallback events (query, filters, timestamp).
4. Test with queries that are guaranteed to fail analytics: a product that doesn't exist in the data, a country name in a different format than what's stored, a date range with no matching rows.
5. Verify the fallback returns a useful RAG answer instead of an empty response.
6. Optionally expose `"fallback": bool` in the `/chat` response for frontend use.

---

## V2 — Quality upgrades

V2 adds three independent improvements. They can be built in any order, but the sequence below is recommended because each one is easier to evaluate once the previous is stable.

---

### V2.1 — Multi-query retrieval

**What it solves.** A single rewritten query often misses relevant chunks because it only covers one phrasing of the information need. "Warranty policy" will retrieve chunks that contain the word "warranty" but miss chunks that say "guarantee period" or "coverage terms". Multi-query expansion generates several phrasings, searches ChromaDB with each, unions the candidate pools, deduplicates, then reranks the full pool in one pass.

**Changes required.** Add `expand()` to `retrieval/query_rewriter.py`. Modify the RAG path and hybrid path in `app.py` to use it instead of `rewrite()`.

**`query_rewriter.py` — add `expand()`:**

```python
def expand(query: str) -> list[str]:
    system = (
        "You are a search query expander. Given a user question, generate "
        f"{MULTI_QUERY_COUNT} alternative search queries that approach the same "
        "information need from different angles. Use synonyms, related terms, and "
        "different phrasings. Include the original question as the first entry. "
        'Return JSON only: {"queries": ["...", "...", "..."]}'
    )
    response = llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": query}],
        response_format={"type": "json_object"},
        max_tokens=200
    )
    data = json.loads(response)
    queries = data.get("queries", [query])
    return queries[:MULTI_QUERY_COUNT]   # cap in case the model returns more
```

**Updated RAG path in `app.py`:**

```python
sub_queries = query_rewriter.expand(message)

all_candidates = []
for q in sub_queries:
    results = vector_store.search(q, top_k=MULTI_QUERY_TOP_K)
    all_candidates.extend(results)

# Deduplicate by chunk ID — keep first occurrence (highest similarity score)
seen_ids = set()
unique_candidates = []
for chunk in all_candidates:
    if chunk["id"] not in seen_ids:
        seen_ids.add(chunk["id"])
        unique_candidates.append(chunk)

# Rerank the full deduplicated pool against the original question
top_chunks = reranker.rerank(message, unique_candidates, top_k=RERANKER_TOP_K)
```

Important: rerank against the **original user message**, not any of the sub-queries. The reranker's job is to score relevance to what the user actually asked, not to what the expander generated.

**Token cost.** The expansion call costs roughly 100–150 tokens. Each additional search is fast (local embedding + ChromaDB). The reranker runs on a larger candidate pool (up to `MULTI_QUERY_COUNT × MULTI_QUERY_TOP_K` before dedup) but it is local inference so latency stays manageable.

**Evaluating improvement.** Before shipping V2.1, run a small evaluation. Take 20 questions where you know the correct answer exists in the documents. Compare RAG V1 (single rewrite) against V2.1 (multi-query) by checking whether the correct chunk appears in the top 5 for each. If multi-query doesn't improve this score for your specific corpus, the added latency may not be worth it.

---

### V2.2 — Improved natural-language filter extraction

**What it solves.** V1's filter extractor works for simple, explicit queries ("total sales in Tunisia in March") but fails on relative dates ("last month", "this quarter", "past 3 months"), multi-value exclusions ("excluding cancelled and returned"), ambiguous column references, and queries about multiple DataFrames at once.

**Changes required.** Improve the system prompt in `analytics/filters.py`. No code changes needed — this is entirely prompt engineering.

**Add date resolution.** Inject today's date into the extraction prompt and instruct the model to resolve all relative date expressions before returning:

```python
from datetime import date

def extract_filters(query: str) -> dict:
    today = date.today().isoformat()
    schema_description = dataframe_manager.get_schema_description()  # column names per DataFrame

    system = f"""
You are a data filter extractor for a business analytics chatbot.
Today's date is {today}.

Available DataFrames:
{schema_description}

Extract a filter object from the user's query. Rules:
1. Resolve all relative date expressions to absolute ISO dates using today's date.
   - "last month" → first and last day of the previous calendar month
   - "this quarter" → first day of current quarter to today
   - "past 3 months" → date 90 days ago to today
   - "this year" → {date.today().year}-01-01 to today
2. All string values must be lowercase to match normalized column values.
3. Use null for any field not mentioned in the query.
4. "exclude_status" is always an array, even for a single value.
5. If the user asks for a ranking or top-N, use "group_by" and "calculation": "group_by".

Return valid JSON only using this schema:
{{"dataframe": string, "filters": {{"product": null, "country": null, "date_start": "YYYY-MM-DD or null",
"date_end": "YYYY-MM-DD or null", "exclude_status": [], "group_by": null}},
"calculation": "sum|mean|count|group_by", "target_column": string}}

Examples:
User: "Total revenue last month excluding cancelled orders"
{{"dataframe": "sales_2026", "filters": {{"product": null, "country": null,
"date_start": "{(date.today().replace(day=1) - pd.DateOffset(months=1)).strftime('%Y-%m-01')}",
"date_end": "{(date.today().replace(day=1) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')}",
"exclude_status": ["cancelled"], "group_by": null}}, "calculation": "sum", "target_column": "revenue"}}

User: "Average order value in France excluding cancelled and returned"
{{"dataframe": "sales_2026", "filters": {{"product": null, "country": "france",
"date_start": null, "date_end": null, "exclude_status": ["cancelled", "returned"],
"group_by": null}}, "calculation": "mean", "target_column": "revenue"}}

User: "Top 5 countries by laptop sales this year"
{{"dataframe": "sales_2026", "filters": {{"product": "laptop", "country": null,
"date_start": "{date.today().year}-01-01", "date_end": "{date.today().isoformat()}",
"exclude_status": [], "group_by": "country"}}, "calculation": "group_by", "target_column": "revenue"}}
"""
    response = llm_client.chat(
        system=system,
        messages=[{"role": "user", "content": query}],
        response_format={"type": "json_object"},
        max_tokens=300
    )
    return fill_filter_defaults(json.loads(response))
```

**`get_schema_description()` in `dataframe_manager.py`:**

```python
def get_schema_description() -> str:
    lines = []
    for name, df in registry.items():
        cols = ", ".join(df.columns.tolist())
        lines.append(f"- {name}: {cols}")
    return "\n".join(lines)
```

This injects live column names so the model always knows what's actually available — no hardcoding column names in the prompt.

**V2.2 build checklist:**

1. Add `get_schema_description()` to `dataframe_manager.py`.
2. Update the `extract_filters()` system prompt as above.
3. Add the `today` injection and relative date instructions.
4. Test with 20 queries including: "last month", "this quarter", "past 90 days", "excluding cancelled and returned", "top 5 by revenue", and one query per available DataFrame.
5. Verify every output has all required fields (no missing keys).
6. Check that date fields are in ISO format, not natural language.

---

### V2.3 — Conversation summary memory

**What it solves.** V1's memory injects the last 10 raw messages into every LLM call. At an average of 150 tokens per message, that is 1,500 tokens of context overhead per call — and it grows proportionally with message length. After 10 turns, earlier context is dropped entirely, so the chatbot forgets the user mentioned a specific product or country several turns ago.

Summary memory replaces the growing message list with a single compressed summary of the conversation so far. The summary is updated after every turn with a small LLM call and stays within a fixed token budget regardless of conversation length.

**Changes required.** Extend `memory/conversation_store.py` with summary functions. Update `app.py` to call them.

**`conversation_store.py` — V2.3 additions:**

```python
summaries: dict[str, str] = {}

def update_summary(session_id: str, user_msg: str, assistant_msg: str):
    existing = summaries.get(session_id, "No prior conversation.")
    prompt = (
        f"Existing summary:\n{existing}\n\n"
        f"New exchange:\nUser: {user_msg}\nAssistant: {assistant_msg}\n\n"
        "Write an updated summary of the full conversation in 3 to 5 sentences. "
        "Include: the user's main topics and questions, any specific entities mentioned "
        "(product names, countries, date ranges, metrics), and conclusions already reached. "
        "Be factual and concise. Return only the summary text."
    )
    new_summary = llm_client.chat(
        system="You are a conversation summarizer.",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=SUMMARY_MAX_TOKENS
    )
    summaries[session_id] = new_summary

def get_summary(session_id: str) -> str:
    return summaries.get(session_id, "")

def clear_summary(session_id: str):
    summaries.pop(session_id, None)
```

**`app.py` — update the query handler:**

Replace the `get_context()` call with summary injection:

```python
# Build context for LLM calls
summary = memory.get_summary(session_id)
summary_context = f"Conversation summary:\n{summary}" if summary else ""

# Pass to LLM as a system-level injection, not as message history
# Modify the system prompt of the answer synthesis call:
full_system = f"{RAG_SYSTEM_PROMPT}\n\n{summary_context}"
```

At the end of every turn, after generating the answer, call:

```python
memory.update_summary(session_id, message, answer)
```

This runs asynchronously after the response is sent so it does not add latency to the user-facing call. In FastAPI, use `BackgroundTasks`:

```python
from fastapi import BackgroundTasks

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    # ... generate answer ...
    background_tasks.add_task(memory.update_summary, request.session_id, request.message, answer)
    return {"answer": answer, "route": route, "sources": sources}
```

**Token comparison.**

| Approach | Tokens per call (10-turn session) | Tokens per call (50-turn session) |
|---|---|---|
| V1: last 10 raw messages | ~1,500 | ~1,500 (drops older turns) |
| V2.3: summary | ~300 | ~300 (fixed regardless of length) |

The summary update call itself costs ~200–400 tokens (existing summary in + new exchange in + summary out). At GPT-4o mini's pricing this is negligible. The net saving per long conversation is significant.

**V2.3 build checklist:**

1. Add `summaries` dict and `update_summary()`, `get_summary()`, `clear_summary()` to `conversation_store.py`.
2. Update `app.py` to inject the summary into the system prompt instead of passing raw message history.
3. Use `BackgroundTasks` to run the summary update after responding.
4. Test across a 20-turn conversation. After turn 15, ask the chatbot to recall something mentioned in turn 3 — it should succeed.
5. Verify the summary stays under `SUMMARY_MAX_TOKENS` and doesn't grow unboundedly.
6. Add `GET /session/{session_id}/summary` endpoint for debugging — returns the current summary for a session.

---

## Complete requirements.txt

```
# Web framework
fastapi
uvicorn[standard]

# LLM
openai>=1.0.0
python-dotenv

# Validation
pydantic

# Document processing
langchain
langchain-community
pypdf
python-docx

# Embeddings and reranking (local models)
sentence-transformers

# Vector database
chromadb>=0.4

# Structured data
pandas
numpy
openpyxl
```

---

## Complete system prompts reference

All prompts in one place for easy tuning.

**Query router:**
```
You are a query classifier for a business intelligence chatbot.
Classify the user's message as one of:
- "rag": questions about policies, product descriptions, procedures, text-based knowledge
- "analytics": questions asking for numbers, totals, averages, rankings, comparisons, or trends
- "hybrid": questions requiring both a number AND explanatory text

For "analytics" and "hybrid", also extract the filter object.
Return JSON: {"route": "rag|analytics|hybrid", "filters": {filter object} or null}
Return valid JSON only. No explanation.
```

**Query rewriter (V1):**
```
You are a search query optimizer. Rewrite the user's question as a clean, specific search query
that would retrieve relevant document chunks. Remove conversational filler. Be specific.
Return only the rewritten query string, nothing else.
```

**Query expander (V2.1):**
```
You are a search query expander. Given a user question, generate 4 alternative search queries
that approach the same information need from different angles. Include the original question first.
Use synonyms, related terms, and different phrasings. Cover aspects the original may have missed.
Return JSON only: {"queries": ["...", "...", "...", "..."]}
```

**Answer synthesis — RAG:**
```
You are a helpful business assistant. Answer the user's question using only the context provided.
Cite the source document when relevant. If the context does not contain the answer, say
"I don't have information about that in the available documents." Do not guess or invent facts.
```

**Answer synthesis — analytics:**
```
You are a data analyst assistant. You have been given exact figures calculated from the business database.
Present these numbers clearly and interpret them briefly. Do not invent or estimate any figures.
If the data seems incomplete or the question cannot be fully answered from the available data, say so.
```

**Answer synthesis — hybrid:**
```
You are a business intelligence assistant. You have been given both calculated data figures
and relevant document context. Use both to give a complete answer.
Lead with the numbers, then add the relevant context. Do not invent any figures or facts.
```

**Memory summarizer (V2.3):**
```
You are a conversation summarizer. Write a concise summary of the conversation so far.
Include: the user's main topics and questions, specific entities mentioned (product names,
countries, date ranges, metrics), and any conclusions or answers already given.
Be factual and brief. 3 to 5 sentences maximum.
```

---

## Common failure points and fixes

**Filter extraction returns partial JSON.** Always use `response_format={"type": "json_object"}` and validate with `fill_filter_defaults()` before passing to Pandas. A missing key causes a KeyError that bypasses the fallback and crashes the request.

**e5 model prefix omitted.** Prepend `"query: "` to query strings and `"passage: "` to chunk texts before encoding with multilingual-e5-large. Without this, similarity scores are systematically wrong and retrieval quality drops significantly. This is easy to forget and does not cause an error — it silently degrades results.

**ChromaDB returns fewer results than `top_k`.** If the collection has fewer chunks than `top_k`, ChromaDB returns what it has without error. The `min(top_k, collection.count())` guard in `vector_store.py` prevents the query from failing. The reranker handles empty input gracefully — always check `if not chunks: return []`.

**Date parsing fails on non-standard formats.** Wrap every `pd.to_datetime()` call in try/except inside `apply_filters()`. Log the failure, skip that particular filter, and continue — a partial result is better than a 500 error.

**Router classifies "analytics" when "hybrid" is correct.** GPT-4o mini sometimes returns "analytics" for queries like "top-selling laptops" where the user implicitly wants warranty and spec information too. Add a post-processing rule in `query_router.py`: if the route is "analytics" and the query contains any product names that also appear as document keywords in ChromaDB, upgrade the route to "hybrid". This can be approximated cheaply with a keyword list derived from your product catalog.

**Summary memory drifts on long conversations.** After 30+ turns, the summary can start to lose early details. Mitigate this by injecting today's date and session length into the summarizer prompt: "This is turn {n} of the conversation. Prioritize recent context but preserve key entities from earlier turns."

**Session memory lost on restart.** Both the raw message store and the summary store are in-memory dicts in V1 and V2. On server restart, all sessions are wiped. For production, replace both dicts with a Redis hash store or SQLite table. The function signatures (`add_message`, `get_context`, `update_summary`, `get_summary`) stay identical — only the storage backend changes.

**Multi-query expansion returns more sub-queries than requested.** Cap the array with `queries[:MULTI_QUERY_COUNT]` after parsing. GPT-4o mini occasionally generates 5 or 6 despite the instruction — the cap prevents unnecessary ChromaDB calls.

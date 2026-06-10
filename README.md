# 🤖 Vilavi Chatbot

> A business intelligence chatbot combining Retrieval-Augmented Generation (RAG) with structured data analytics. Built with FastAPI, LangChain, and Streamlit.

![Version](https://img.shields.io/badge/version-2.7.5-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Last Updated](https://img.shields.io/badge/updated-June%202026-lightgrey?style=flat-square)

---

## 📋 Table of Contents

- [Overview](#overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [⚙️ Configuration](#️-configuration)
- [💬 Usage](#-usage)
- [📡 API Reference](#-api-reference)
- [📁 Project Structure](#-project-structure)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## Overview

Vilavi Chatbot enables natural language querying over both unstructured documents and structured data. It automatically routes questions to the right engine — semantic document search, pandas-powered analytics, or a hybrid of both — and returns answers in French with inline source citations.

---

## ✨ Features

### 📄 Document Intelligence (RAG)

| Capability | Details |
|---|---|
| Multi-format ingestion | PDF, DOCX, and TXT files |
| Semantic search | ChromaDB vector database with multilingual E5 embeddings |
| Intelligent reranking | BGE reranker for improved relevance scoring |
| Contextual compression | Refines retrieved chunks for higher precision |
| Query condensation | Resolves pronouns and references from conversation history |
| Source citations | Every factual claim backed by an inline citation |
| Language | All responses generated in French 🇫🇷 |

### 📊 Analytics Engine

| Capability | Details |
|---|---|
| Structured data processing | Ingests CSV and Excel files |
| LLM-generated Pandas code | Translates natural language into data queries |
| Schema-aware | Understands column names, data types, and relationships |
| Auto error correction | Retries failed queries with corrected code (up to 3 attempts) |
| Data normalization | Automatic column cleaning, date parsing, type conversion |
| Smart keyword search | Intelligently searches across multiple text columns |

### 🧠 Intelligent Routing

| Capability | Details |
|---|---|
| Query classification | Auto-routes to RAG, Analytics, or Hybrid mode |
| Parallel hybrid execution | Analytics and RAG retrieval run simultaneously |
| Semantic caching | Caches similar queries at 98% similarity threshold |
| Conversation memory | Maintains context across follow-ups (configurable window) |

### 🎨 Frontend

| Capability | Details |
|---|---|
| Modern UI | Glassmorphism design with gradient backgrounds |
| Real-time chat | Message history with expandable source citations |
| Data tables | Interactive display for analytics results |
| Health indicator | Live backend connection status |
| Memory reset | Clear conversation history with one click |

---

## 🏗️ Architecture

How a user query flows through the system:

```
                         ┌─────────────────────────────────────┐
                         │         Streamlit Frontend           │
                         │  Chat UI · Source viewer · Tables    │
                         └──────────────────┬──────────────────┘
                                            │ POST /chat
                         ┌──────────────────▼──────────────────┐
                         │           FastAPI Backend            │
                         │                                      │
                         │   ┌────────────────────────────┐    │
                         │   │     Semantic Cache (98%)    │    │
                         │   └──────────────┬─────────────┘    │
                         │                  │ cache miss        │
                         │   ┌──────────────▼─────────────┐    │
                         │   │       Query Router          │    │
                         │   └───────┬──────────┬──────────┘    │
                         │           │          │               │
                         │    ┌──────▼──┐  ┌───▼──────┐        │
                         │    │   RAG   │  │Analytics │        │
                         │    │ Pipeline│  │ Engine   │        │
                         │    └──────┬──┘  └───┬──────┘        │
                         │           │          │               │
                         │    ┌──────▼──┐  ┌───▼──────┐        │
                         │    │ChromaDB │  │  Pandas  │        │
                         │    │Vectors  │  │  + CSV   │        │
                         │    └─────────┘  └──────────┘        │
                         │                                      │
                         │         OpenRouter LLM               │
                         │       (GPT-4o-mini default)          │
                         └──────────────────────────────────────┘
```

### Query routing logic

```
User query
    │
    ├─── Contains data keywords? ──► Analytics Engine
    │         (sales, count, total…)       │
    │                                      │
    ├─── Contains doc keywords?  ──► RAG Pipeline
    │         (policy, terms, explain…)    │
    │                                      │
    └─── Both?  ─────────────────► Hybrid (parallel)
                                           │
                                    Merge & respond
```

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────────┐
│                     Frontend                        │
│          Streamlit · Requests · Pandas              │
├─────────────────────────────────────────────────────┤
│                      API Layer                      │
│                FastAPI (async REST)                 │
├──────────────────────┬──────────────────────────────┤
│     RAG Pipeline     │       Analytics Engine       │
│  LangChain           │  Pandas · NumPy              │
│  ChromaDB            │  LLM code generation         │
│  E5 Embeddings       │  Auto error correction       │
│  BGE Reranker        │                              │
├──────────────────────┴──────────────────────────────┤
│                     Models                          │
│  LLM: OpenRouter (GPT-4o-mini)                      │
│  Embeddings: intfloat/multilingual-e5-large         │
│  Reranker:   BAAI/bge-reranker-base                 │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- `pip`
- An [OpenRouter](https://openrouter.ai) API key

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd chatbot2
```

### Step 2 — Set up the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL_NAME=openai/gpt-4o-mini
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=20
TOP_N=5
```

Place your data files in the correct directories:

```
backend/data/
├── 📁 documents/
│   ├── 📂 pdf/          ← PDF files
│   ├── 📂 docx/         ← Word documents
│   └── 📂 txt/          ← Plain text files
└── 📁 csv/              ← CSV or Excel files for analytics
```

Run ingestion, then start the server:

```bash
python ingest.py
python app.py
# ✅ Backend running at http://0.0.0.0:8000
```

### Step 3 — Set up the frontend

```bash
cd ../frontend_streamlit
pip install -r requirements.txt

export API_URL=http://127.0.0.1:8000   # Windows: set API_URL=...
streamlit run app.py
# ✅ Frontend running at http://localhost:8501
```

---

## ⚙️ Configuration

Key parameters in `backend/config.py`. All can be overridden via `.env`.

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Token size for document chunks |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `VECTOR_SEARCH_TOP_K` | `20` | Documents retrieved before reranking |
| `RERANKER_TOP_K` | `5` | Documents retained after reranking |
| `MEMORY_SIZE` | `10` | Conversation turns held in memory |
| `MULTI_QUERY_COUNT` | `4` | Number of query expansions generated |

### How chunk parameters affect quality

```
 CHUNK_SIZE  ──────────────────────────────────────────────►
 
 Small (500)   │ Fast · cheap · may lose context
               │
 Medium (1000) │ ✅ Recommended default — balanced precision
               │
 Large (2000)  │ Rich context · slower · higher cost

 CHUNK_OVERLAP ──────────────────────────────────────────────►
 
 Low (50)    │ Faster ingestion · may miss boundary content
 High (300)  │ ✅ Better continuity · larger index size
```

---

## 💬 Usage

### Document search 📄

```
"What are the warranty terms?"
"Explain the return policy."
"What payment options are available?"
```

### Data analytics 📊

```
"Show total sales by region."
"List all customers who signed up in 2023."
"What is the average order value?"
```

### Hybrid queries 🔀

```
"What is the return policy, and how many returns were processed last month?"
"Show me the warranty terms alongside related claim statistics."
```

---

## 📡 API Reference

### Endpoints at a glance

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System status and document count |
| `POST` | `/chat` | Send a message and receive a response |
| `POST` | `/ingest` | Trigger document re-indexing |
| `POST` | `/clear-memory` | Reset conversation history |

### `GET /health`

```bash
curl http://localhost:8000/health
```

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the warranty terms?"}'
```

### `POST /ingest`

```bash
curl -X POST http://localhost:8000/ingest
```

### `POST /clear-memory`

```bash
curl -X POST http://localhost:8000/clear-memory
```

---

## 📁 Project Structure

```
chatbot2/
├── 📦 backend/
│   ├── app.py              # FastAPI application entry point
│   ├── config.py           # Configuration and environment variables
│   ├── rag.py              # RAG pipeline implementation
│   ├── analytics.py        # Analytics engine
│   ├── router.py           # Query routing logic
│   ├── ingest.py           # Document ingestion script
│   ├── models.py           # ML model loading and utilities
│   ├── requirements.txt
│   ├── 📂 data/
│   │   ├── documents/      # Unstructured document storage
│   │   │   ├── pdf/
│   │   │   ├── docx/
│   │   │   └── txt/
│   │   └── csv/            # Structured data for analytics
│   └── 🗄️ chroma_db/       # Persisted vector database
├── 🎨 frontend_streamlit/
│   ├── app.py              # Streamlit UI
│   └── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

### 🟢 Short-term

- [ ] User authentication and session management
- [ ] Real-time document upload through the UI
- [ ] Charts and graphs for analytics results
- [ ] API rate limiting for production
- [ ] Docker images and CI/CD pipeline

### 🟡 Medium-term

- [ ] Hybrid keyword + semantic search
- [ ] Document versioning
- [ ] Response streaming for better UX
- [ ] User feedback collection for response quality improvement
- [ ] Webhook integrations for external systems

### 🔵 Long-term

- [ ] Multi-modal support (images, audio, video)
- [ ] Advanced RAG techniques (graph RAG, hierarchical RAG)
- [ ] Domain-specific model fine-tuning
- [ ] Horizontal scaling for large deployments
- [ ] Plugin system for third-party extensions

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change, then submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a pull request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🆘 Support

For questions or bug reports, please [open an issue](../../issues) in the repository.

---

<div align="center">

Made with ❤️ · **Version 2.7.5** · Last updated June 2026

</div>

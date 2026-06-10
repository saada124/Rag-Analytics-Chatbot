# Vilavi Chatbot

A sophisticated business intelligence chatbot that combines Retrieval-Augmented Generation (RAG) with advanced analytics capabilities. Built with FastAPI, LangChain, and Streamlit, it provides intelligent document search, data analysis, and natural language querying for enterprise data.

## 🚀 Features

### Document Intelligence (RAG)
- **Multi-format Document Ingestion**: Supports PDF, DOCX, and TXT files
- **Vector Database**: ChromaDB for efficient semantic search
- **Advanced Embeddings**: E5 multilingual embeddings for cross-lingual support
- **Intelligent Reranking**: BGE reranker for improved relevance scoring
- **Contextual Compression**: Refines search results for better precision
- **Query Condensation**: Resolves pronouns and references from conversation history
- **Source Citation**: Provides inline citations for all factual claims
- **French Language Support**: All responses are generated in French

### Analytics Engine
- **Structured Data Processing**: Ingests CSV and Excel files
- **LLM-Powered Query Generation**: Automatically generates Pandas code for data analysis
- **Schema-Aware**: Understands data structure and column relationships
- **Automatic Error Correction**: Retries failed queries with corrected code (up to 3 attempts)
- **Data Normalization**: Automatic column cleaning, date parsing, and type conversion
- **Smart Keyword Search**: Intelligently searches across multiple text columns
- **Result Formatting**: Converts results to JSON-friendly formats with proper type handling

### Intelligent Routing
- **Query Classification**: Automatically routes queries to RAG, Analytics, or Hybrid modes
- **Parallel Execution**: Hybrid queries run analytics and RAG retrieval simultaneously
- **Semantic Caching**: Caches similar queries to improve response time (98% similarity threshold)
- **Conversation Memory**: Maintains context for follow-up questions (configurable size)

### Modern Frontend
- **Glassmorphism UI**: Beautiful, modern interface with gradient backgrounds
- **Chat Interface**: Real-time chat with message history
- **Source Display**: Expandable sections showing document sources
- **Data Visualization**: Interactive data tables for analytics results
- **Health Monitoring**: Real-time backend connection status
- **Memory Management**: Clear conversation history with one click

### Backend API
- **FastAPI Framework**: High-performance async API
- **REST Endpoints**: Clean RESTful API design
- **Health Checks**: Monitor system status and document count
- **Ingestion Control**: Manual trigger for document re-indexing
- **Memory Management**: Clear conversation memory endpoint
- **Lifespan Management**: Automatic data loading on startup

## 🛠️ Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Open-source vector database
- **PyTorch**: Deep learning framework for model execution
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Sentence Transformers**: State-of-the-art embeddings

### Models
- **LLM**: OpenRouter (GPT-4o-mini by default)
- **Embeddings**: intfloat/multilingual-e5-large
- **Reranker**: BAAI/bge-reranker-base

### Frontend
- **Streamlit**: Python framework for ML apps
- **Requests**: HTTP library for API calls
- **Pandas**: Data visualization

## 📋 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- OpenRouter API key

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd chatbot2
```

2. **Navigate to backend directory**
```bash
cd backend
```

3. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Configure environment variables**
Create a `.env` file in the backend directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL_NAME=openai/gpt-4o-mini
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=20
TOP_N=5
```

6. **Prepare data directories**
Place your documents in the appropriate directories:
- `backend/data/documents/pdf/` - PDF files
- `backend/data/documents/docx/` - Word documents
- `backend/data/documents/txt/` - Text files
- `backend/data/csv/` - CSV/Excel files for analytics

7. **Run ingestion**
```bash
python ingest.py
```

8. **Start the backend server**
```bash
python app.py
```
The backend will start on `http://0.0.0.0:8000`

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend_streamlit
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API URL**
Set the `API_URL` environment variable:
```bash
export API_URL=http://127.0.0.1:8000  # On Windows: set API_URL=http://127.0.0.1:8000
```

4. **Start the frontend**
```bash
streamlit run app.py
```
The frontend will open in your browser at `http://localhost:8501`

## 📖 Usage

### Document Search
Ask questions about your documents:
- "What are the warranty terms?"
- "Explain the return policy"
- "What are the payment options?"

### Data Analytics
Query your structured data:
- "Show me total sales by region"
- "List all customers who signed up in 2023"
- "What's the average order value?"

### Hybrid Queries
Combine document context with data analysis:
- "What's the return policy and how many returns were processed last month?"
- "Show me the warranty terms and related claim statistics"

### API Usage

**Health Check**
```bash
curl http://localhost:8000/health
```

**Chat**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the warranty terms?"}'
```

**Ingest Documents**
```bash
curl -X POST http://localhost:8000/ingest
```

**Clear Memory**
```bash
curl -X POST http://localhost:8000/clear-memory
```

## 🔧 Configuration

Key configuration parameters in `backend/config.py`:

- `CHUNK_SIZE`: Document chunk size for vectorization (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `VECTOR_SEARCH_TOP_K`: Number of documents to retrieve (default: 20)
- `RERANKER_TOP_K`: Number of documents after reranking (default: 5)
- `MEMORY_SIZE`: Conversation history size (default: 10)
- `MULTI_QUERY_COUNT`: Number of query expansions (default: 4)

## 📁 Project Structure

```
chatbot2/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── config.py           # Configuration settings
│   ├── rag.py              # RAG implementation
│   ├── analytics.py        # Analytics engine
│   ├── router.py           # Query routing logic
│   ├── ingest.py           # Document ingestion
│   ├── models.py           # ML models and utilities
│   ├── requirements.txt    # Backend dependencies
│   ├── data/
│   │   ├── documents/      # Document storage
│   │   │   ├── pdf/
│   │   │   ├── docx/
│   │   │   └── txt/
│   │   └── csv/            # Structured data
│   └── chroma_db/          # Vector database
├── frontend_streamlit/
│   ├── app.py              # Streamlit frontend
│   └── requirements.txt    # Frontend dependencies
└── README.md
```

## 🚧 Future Enhancements

### Short-term Improvements
- **User Authentication**: Add user login and session management
- **Multi-language Support**: Extend beyond French to support multiple languages
- **Real-time Document Upload**: Allow users to upload documents through the UI
- **Advanced Analytics Visualization**: Add charts and graphs for data insights
- **API Rate Limiting**: Implement rate limiting for production deployment
- **Docker Containerization**: Create Docker images for easy deployment
- **CI/CD Pipeline**: Set up automated testing and deployment

### Medium-term Enhancements
- **Hybrid Search**: Combine keyword search with semantic search
- **Document Versioning**: Track and manage document versions
- **User Feedback Collection**: Allow users to rate responses for improvement
- **Performance Monitoring**: Add logging and metrics for system health
- **Response Streaming**: Implement streaming responses for better UX
- **Webhook Integrations**: Support external system integrations
- **Mobile App**: Develop a mobile application

### Long-term Vision
- **Multi-modal Support**: Handle images, audio, and video content
- **Advanced RAG Techniques**: Implement graph RAG, hierarchical RAG
- **Custom Model Fine-tuning**: Fine-tune models on domain-specific data
- **Collaborative Features**: Enable sharing and collaboration on queries
- **Advanced Security**: Add encryption, audit logs, and compliance features
- **Scalability**: Implement horizontal scaling for large deployments
- **Plugin System**: Allow third-party extensions and integrations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For support and questions, please open an issue in the repository.

---

**Version**: 2.7.5  
**Last Updated**: June 2026

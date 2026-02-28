# Enterprise RAG Demo

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and Claude. Upload documents, build a knowledge base, and ask questions — the system retrieves relevant context and generates accurate, grounded answers.

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────┐
│  Upload   │───>│  Parse &     │───>│  Embed &   │───>│ ChromaDB│
│  (API/UI) │    │  Chunk       │    │  Store     │    │  Vector │
└──────────┘    └──────────────┘    └────────────┘    │  Store  │
                                                       └────┬────┘
┌──────────┐    ┌──────────────┐    ┌────────────┐         │
│  Query   │───>│  Embed Query │───>│  Retrieve  │<────────┘
│  (API/UI) │    │              │    │  Top-K     │
└──────────┘    └──────────────┘    └─────┬──────┘
                                          │
                                    ┌─────▼──────┐    ┌─────────┐
                                    │  Augment   │───>│  Claude  │
                                    │  Prompt    │    │  LLM    │
                                    └────────────┘    └────┬────┘
                                                          │
                                                    ┌─────▼──────┐
                                                    │  Response  │
                                                    │  + Sources │
                                                    └────────────┘
```

## Features

- **Document Ingestion** — Upload PDF, DOCX, TXT, Markdown, CSV files
- **Smart Chunking** — Recursive character splitting with configurable size/overlap
- **Vector Search** — Cosine similarity search via ChromaDB with score thresholds
- **RAG Generation** — Context-augmented answers powered by Claude
- **Web UI** — Clean, responsive interface with dark mode
- **REST API** — Full OpenAPI/Swagger docs at `/docs`
- **JWT Auth** — Optional token-based authentication
- **Prometheus Metrics** — Request latency, query counts, ingestion stats
- **Structured Logging** — JSON-formatted logs via structlog

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure (optional)

```bash
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY
```

The system works without an API key — it returns the most relevant document chunks directly instead of generating an LLM answer.

### 3. Run the server

```bash
python -m src.api.main
```

Or use the entry point:

```bash
rag-server
```

Visit **http://localhost:8000** for the web UI, or **http://localhost:8000/docs** for the Swagger API docs.

### 4. Load sample data

Upload the included sample documents via the UI, or use the API:

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@data/sample_docs/company_handbook.txt"

curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@data/sample_docs/product_faq.txt"

curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@data/sample_docs/api_docs.txt"
```

### 5. Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the PTO policy?", "top_k": 5}'
```

## Project Structure

```
src/
├── api/              # FastAPI routes and application
│   ├── main.py       # App factory, middleware, startup
│   ├── routes.py     # REST API endpoints
│   └── ui_routes.py  # Frontend page routes
├── core/
│   └── config.py     # Settings from YAML + env vars
├── ingestion/        # Document processing pipeline
│   ├── parser.py     # PDF, DOCX, TXT extraction
│   ├── chunker.py    # Text splitting strategies
│   └── pipeline.py   # End-to-end ingest orchestration
├── retrieval/        # Search and generation
│   ├── embeddings.py # Sentence-transformer embeddings
│   ├── vector_store.py # ChromaDB wrapper
│   └── rag_engine.py # Retrieve → augment → generate
├── enterprise/       # Production features
│   ├── auth.py       # JWT authentication
│   ├── logging.py    # Structured logging setup
│   └── monitoring.py # Prometheus metrics + middleware
└── models/
    └── schemas.py    # Pydantic request/response models
```

## Configuration

Edit `config/settings.yaml` or use environment variables:

| Setting | Env Var | Default |
|---------|---------|---------|
| LLM Model | `ANTHROPIC_API_KEY` | — |
| Chunk Size | — | 512 |
| Chunk Overlap | — | 50 |
| Top-K Results | — | 5 |
| Score Threshold | — | 0.3 |
| Auth Enabled | `RAG_AUTH_ENABLED` | false |

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Docker

```bash
docker build -t enterprise-rag .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key enterprise-rag
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/documents/upload` | Upload a document |
| `GET` | `/api/v1/documents` | List all documents |
| `GET` | `/api/v1/documents/{id}` | Get document details |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `POST` | `/api/v1/query` | Ask a question (RAG) |
| `GET` | `/api/v1/collections` | List collections |
| `POST` | `/api/v1/auth/token` | Get JWT token |
| `GET` | `/api/v1/metrics` | Prometheus metrics |

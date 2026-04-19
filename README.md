# AI Customer Support Multi-Agent System

A GenAI-powered multi-agent system that lets a customer support executive query structured customer data (SQL) and unstructured policy documents (RAG) through a conversational Streamlit chat UI — all routed intelligently via **Claude MCP tool_use**.

---

## How It Works

The system has two agents behind one chat interface:

- **SQL Agent** — answers questions about customers and tickets by generating SQLite queries with Claude Haiku and executing them against a local database
- **RAG Agent** — answers policy questions by searching indexed PDF documents with semantic similarity and generating answers with Claude Haiku

**Routing** is done by Claude itself via MCP tool selection. When a user sends a message, Claude receives 6 tool definitions (one per category card) and picks the right one based on intent — no keyword matching or separate classifier needed.

```
User Query
    │
    ▼
Claude Haiku (MCP tool_use)
    │  selects one of 6 tools:
    │  search_customers / get_support_tickets / get_billing_and_plans
    │  search_refund_policy / search_cancellation_terms / search_service_agreement
    │
    ├──► SQL Agent ──► SQLite (data/support.db) ──► Claude Haiku formats result
    │
    └──► RAG Agent ──► ChromaDB similarity search ──► Claude Haiku answers from context
```

---

## Architecture

```
ai-support-agent/
├── agents/
│   ├── sql_agent.py        # NL → SQL → SQLite → formatted answer
│   └── rag_agent.py        # PDF chunking, ChromaDB indexing, policy Q&A
├── graph/
│   └── router.py           # Claude MCP tool_use routing (6 tools)
├── server/
│   └── mcp_server.py       # FastAPI REST wrapper (POST /query, GET /health, GET /tools)
├── ui/
│   └── app.py              # Streamlit chat UI with hero, category cards, agent badges
├── data/
│   ├── support.db          # SQLite: 20 customers + 20 tickets
│   └── pdfs/               # Place policy PDFs here for indexing
├── chroma_db/              # ChromaDB persisted vector store (auto-created)
├── setup_db.py             # Seeds the SQLite database
├── requirements.txt        # Pinned dependencies
├── Dockerfile
├── docker-compose.yml
└── .env                    # API keys and paths (not committed)
```

---

## Tech Stack

| Component     | Technology                            | Why                                      |
|---------------|---------------------------------------|------------------------------------------|
| LLM           | Claude Haiku (`claude-haiku-4-5`)     | Fast, cost-effective, supports tool_use  |
| Routing       | Claude MCP tool_use                   | Intent-based tool selection, no classifier |
| SQL DB        | SQLite                                | Zero setup, file-based                   |
| Vector DB     | ChromaDB (local persistent)           | No server required, local persist        |
| Embeddings    | `all-MiniLM-L6-v2` (HuggingFace)     | Free, runs locally, no API key needed    |
| UI            | Streamlit                             | Rapid chat UI with file upload           |
| MCP Server    | FastAPI + Uvicorn                     | Thin REST wrapper for production         |
| Containerization | Docker + docker-compose            | Two-service deploy (UI + MCP server)     |

---

## Setup (Local)

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone the repository

```bash
git clone <repo-url>
cd ai-support-agent
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...        
DB_PATH=data/support.db
CHROMA_PERSIST_DIR=./chroma_db
POLICIES_DIR=data/pdfs
HF_HUB_OFFLINE=1                    
TRANSFORMERS_OFFLINE=1
```

> **Important:** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` prevent the embedding model from hanging on network requests. The model is cached locally after the first run.

### 5. Seed the database

```bash
python setup_db.py
```

This creates `data/support.db` with:
- **20 customers** — name, email, subscription plan (free/starter/pro/enterprise), join date
- **20 support tickets** — subject, status (open/in_progress/resolved/closed), date

The script is idempotent — running it again on an existing database is a no-op.

### 6. Run the Streamlit UI

```bash
venv/bin/python -m streamlit run ui/app.py
```

Opens at `http://localhost:8501`

---

## Indexing Policy PDFs

Policy documents must be indexed into ChromaDB before RAG queries work.

**Option A — Sidebar button:**
Place PDFs in `data/pdfs/` and click **"🔄 Index all PDFs"** in the sidebar.

**Option B — Upload directly:**
Use the sidebar file uploader. PDFs up to 10 MB are accepted and indexed automatically.

**Option C — Command line:**
```bash
venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from agents.rag_agent import index_pdfs
print(index_pdfs('data/pdfs'), 'chunks indexed')
"
```

Already-indexed documents are skipped on re-run (deduplication is done by checking ChromaDB metadata). The embedding model (`all-MiniLM-L6-v2`) loads on the first index and stays cached — subsequent queries are fast.

---

## Running the MCP Server (Optional)

The MCP server is a FastAPI REST wrapper around the same routing logic. The Streamlit UI calls it automatically when `MCP_URL` is set; in local dev it falls back to a direct function call.

```bash
venv/bin/python -m uvicorn server.mcp_server:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path      | Description                                      |
|--------|-----------|--------------------------------------------------|
| `GET`  | `/health` | System status — API key set, DB exists, ChromaDB exists |
| `GET`  | `/tools`  | List all 6 MCP tool definitions                  |
| `POST` | `/query`  | `{"message": "your question"}` → `{"response": "...", "agent": "sql"|"rag"}` |

### Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "List all open support tickets"}'
```

```json
{
  "response": "There are 5 open support tickets: ...",
  "agent": "sql"
}
```

---

## Running with Docker

```bash
docker-compose up --build
```

This starts two containers:
- **`streamlit`** — Streamlit UI on port 8501
- **`mcp`** — FastAPI MCP server on port 8000

Both containers share the `data/` and `chroma_db/` directories via Docker volumes. The Streamlit container connects to the MCP server at `http://mcp:8000`.

```bash
# Stop
docker-compose down

# Rebuild after code changes
docker-compose up --build
```

> **Note:** Create your `.env` file before running Docker. The compose file uses `env_file: .env`.

---

## Example Queries

### SQL Agent (structured customer data)

| Query | What it does |
|-------|-------------|
| `Show me all customer profiles` | Lists all 20 customers with their plans |
| `Give me Emma Brown's full profile and ticket history` | JOIN query across customers + tickets |
| `List all open support tickets` | Filters tickets by status = 'open' |
| `How many customers are on the Pro plan?` | COUNT + WHERE query |
| `Show a breakdown of customers by subscription plan` | GROUP BY plan |

### RAG Agent (policy documents)

| Query | What it does |
|-------|-------------|
| `What is the current refund policy?` | Searches indexed PDFs for refund terms |
| `What are the cancellation terms?` | Finds cancellation/termination clauses |
| `Summarise the key points of the service agreement` | Retrieves top chunks + summarizes |

---

## MCP Tool Definitions

The router exposes 6 tools to Claude. Claude picks one based on the user's intent:

| Tool | Routes to | Handles |
|------|-----------|---------|
| `search_customers` | SQL Agent | Customer profiles, names, emails, plan membership |
| `get_support_tickets` | SQL Agent | Open/closed/resolved tickets, support history |
| `get_billing_and_plans` | SQL Agent | Plan breakdown, billing stats, subscription info |
| `search_refund_policy` | RAG Agent | Refund rules, money-back guarantees |
| `search_cancellation_terms` | RAG Agent | Cancellation procedures, termination clauses |
| `search_service_agreement` | RAG Agent | Terms of service, user rights, acceptable use |

---

## UI Features

- **Hero banner** — gradient header with personalized greeting
- **Category cards** — 6 quick-action cards (shown before first message) that pre-fill a suggested query
- **Agent badge pills** — every assistant response shows whether it came from the SQL Agent (blue) or RAG Agent (green)
- **PDF sidebar** — lists indexed documents, "Index all PDFs" button, and file uploader
- **Live metrics** — customer count and open ticket count shown in sidebar
- **New conversation** — resets chat history without reloading the page

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `DB_PATH` | `data/support.db` | Path to the SQLite database |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Directory for ChromaDB persistence |
| `POLICIES_DIR` | `data/pdfs` | Directory scanned for PDFs to index |
| `MCP_URL` | *(empty)* | MCP server URL — if set, UI calls MCP server; otherwise calls router directly |
| `HF_HUB_OFFLINE` | `1` | Set to `1` to prevent HuggingFace network calls |
| `TRANSFORMERS_OFFLINE` | `1` | Set to `1` to prevent transformers network calls |

---

## Troubleshooting

**"Sorry, I couldn't retrieve that information"**
→ Check your `ANTHROPIC_API_KEY` in `.env`. An invalid or expired key causes every LLM call to fail with a 401 error.

**PDF indexing hangs / takes very long**
→ Ensure `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in `.env`. Without these, the embedding model tries to contact HuggingFace Hub and can hang indefinitely.

**"No policy documents are indexed yet"**
→ Place PDF files in `data/pdfs/` and click "🔄 Index all PDFs" in the sidebar. The first index takes ~15–30s while the embedding model loads.

**Customer name not found (e.g. "Ema" instead of "Emma")**
→ The SQL agent applies fuzzy LIKE matching and progressively shortens patterns to handle partial names and typos. Use partial names like "Emma" or "Emma B".

**Database not found**
→ Run `python setup_db.py` to create and seed `data/support.db`.

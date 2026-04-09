import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from graph.router import ask, MCP_TOOLS

app = FastAPI(title="AI Support MCP Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    response: str
    agent: str


class ToolsResponse(BaseModel):
    tools: list


@app.get("/health")
def health():
    db_path = os.getenv("DB_PATH", "data/support.db")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    checks = {
        "api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
        "db_exists": os.path.isfile(db_path),
        "chroma_exists": os.path.isdir(chroma_dir),
    }
    status = "ok" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks}


@app.get("/tools", response_model=ToolsResponse)
def list_tools():
    """List all available MCP tools."""
    return {"tools": [{"name": t["name"], "description": t["description"]} for t in MCP_TOOLS]}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        logger.info("MCP query received: %s", request.message[:80])
        result = ask(request.message)
        return QueryResponse(response=result["output"], agent=result["route"])
    except Exception as e:
        logger.error("MCP query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.mcp_server:app", host="0.0.0.0", port=8000, reload=True)

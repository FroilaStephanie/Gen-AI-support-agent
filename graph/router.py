import os
import logging
from dotenv import load_dotenv
import anthropic

load_dotenv()

logger = logging.getLogger(__name__)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── MCP Tool Definitions ──────────────────────────────────────────────────────
# Claude uses these to decide which tool to call based on the user's query.
# Each tool maps to one of the 6 UI card categories.

MCP_TOOLS = [
    {
        "name": "search_customers",
        "description": (
            "Search and retrieve customer profiles from the database. "
            "Use this for queries about customers, their names, email addresses, "
            "subscription plans, join dates, or when listing all customers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language question about customers. "
                        "Examples: 'show all customers', 'find customers on pro plan', "
                        "'get Alice Chen profile', 'list enterprise customers'"
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_support_tickets",
        "description": (
            "Retrieve support tickets from the database. "
            "Use this for queries about open, closed, in_progress, or resolved tickets, "
            "customer issues, or support history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language question about support tickets. "
                        "Examples: 'list all open tickets', 'show resolved tickets', "
                        "'what issues does customer X have', 'how many tickets are open'"
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_billing_and_plans",
        "description": (
            "Retrieve billing and subscription plan information from the database. "
            "Use this for queries about plans (free, starter, pro, enterprise), "
            "customer counts per plan, upgrades, or billing statistics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language question about billing or subscription plans. "
                        "Examples: 'how many customers on pro plan', "
                        "'list enterprise customers', 'billing breakdown by plan', "
                        "'which plan does Alice have'"
                    ),
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_refund_policy",
        "description": (
            "Search refund policy documents to answer questions about refunds, "
            "money-back guarantees, reimbursements, or return policies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question about refund rules or policies",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_cancellation_terms",
        "description": (
            "Search cancellation terms documents. Use for questions about "
            "cancellation procedures, contract termination, service discontinuation, "
            "or subscription cancellation policies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question about cancellation or termination terms",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_service_agreement",
        "description": (
            "Search service agreement and general terms documents. Use for questions "
            "about terms of service, user rights, obligations, acceptable use, "
            "or general service conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question about the service agreement or terms of service",
                }
            },
            "required": ["query"],
        },
    },
]

# Tools that go to the SQL agent vs RAG agent
_SQL_TOOLS = {"search_customers", "get_support_tickets", "get_billing_and_plans"}
_RAG_TOOLS = {"search_refund_policy", "search_cancellation_terms", "search_service_agreement"}


def _dispatch_tool(tool_name: str, query: str, original_query: str = "") -> tuple[str, str]:
    """Execute the MCP tool selected by Claude. Returns (result_text, agent_type)."""
    if tool_name in _SQL_TOOLS:
        from agents.sql_agent import query_database
        logger.info("Dispatching to SQL agent: %s", tool_name)
        # Pass original_query so ticket detection uses the user's exact words,
        # not the rephrased sub-query Claude extracted for the tool call.
        return query_database(query, original_query=original_query), "sql"

    if tool_name in _RAG_TOOLS:
        from agents.rag_agent import query_policies
        logger.info("Dispatching to RAG agent: %s", tool_name)
        return query_policies(query), "rag"

    logger.warning("Unknown tool requested: %s", tool_name)
    return f"Unknown tool: {tool_name}", "sql"


def ask(query: str) -> dict:
    """
    Main entry point. Routes the user query via Claude MCP tool selection.

    Returns:
        {"output": str, "route": "sql" | "rag"}
    """
    query = query.strip()
    if not query:
        return {"output": "Please enter a question.", "route": "sql"}

    try:
        # Claude selects the appropriate MCP tool based on the query
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=MCP_TOOLS,
            messages=[{"role": "user", "content": query}],
        )

        # Find the tool_use block in the response
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None,
        )

        if tool_block:
            tool_name = tool_block.name
            sub_query = tool_block.input.get("query", query)
            logger.info("MCP tool selected: '%s' for query: '%s...'", tool_name, query[:60])
            result, agent_type = _dispatch_tool(tool_name, sub_query, original_query=query)
            return {"output": result, "route": agent_type}

        # Claude responded with text instead of a tool call — return it directly
        text = next(
            (b.text for b in response.content if hasattr(b, "text")),
            "I couldn't determine how to answer that. Please rephrase your question.",
        )
        logger.info("No tool selected; returning direct text response")
        return {"output": text, "route": "sql"}

    except anthropic.AuthenticationError:
        logger.error("Anthropic API key invalid or expired")
        return {
            "output": "⚠️ API key is invalid or expired. Please update ANTHROPIC_API_KEY in your .env file.",
            "route": "sql",
        }
    except Exception as e:
        logger.error("MCP routing failed for query '%s': %s", query[:60], e, exc_info=True)
        return {"output": "Sorry, something went wrong. Please try again.", "route": "sql"}

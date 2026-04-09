import re
import sqlite3
import os
import logging
from dotenv import load_dotenv
import anthropic

load_dotenv()

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/support.db")
_SQL_BLOCKLIST = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|ATTACH)\b",
    re.IGNORECASE,
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_schema() -> str:
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Run setup_db.py first.")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(customers)")
        cols = cur.fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        schema = f"customers({col_defs})\n  -- customers.id is the primary key (NOT customer_id)\n"

        cur.execute("PRAGMA table_info(tickets)")
        cols = cur.fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        schema += f"tickets({col_defs})\n  -- tickets.customer_id references customers.id"
    return schema


_TICKET_KEYWORDS = re.compile(
    r"\b(ticket|tickets|history|issues?|support history|cases?)\b", re.IGNORECASE
)


def _wants_tickets(question: str) -> bool:
    return bool(_TICKET_KEYWORDS.search(question))


def nl_to_sql(question: str, schema: str, wants_tickets: bool = False) -> str:
    ticket_hint = ""
    if wants_tickets:
        ticket_hint = """
IMPORTANT: This question asks about tickets or history. You MUST use this exact JOIN pattern:
  SELECT c.id, c.name, c.email, c.plan, c.joined_date,
         t.id as ticket_id, t.subject, t.status, t.created_at as ticket_date
  FROM customers c
  LEFT JOIN tickets t ON t.customer_id = c.id
  WHERE LOWER(c.name) LIKE '%<name>%'
Replace <name> with the customer name from the question (lowercase). Do NOT write a query without the JOIN.
"""

    prompt = f"""You are a SQL expert. Given the schema below, write a valid SQLite SQL query to answer the user's question.
Return ONLY the SQL query, nothing else.

Rules:
- ALWAYS use LIKE with wildcards for name searches (e.g. WHERE LOWER(c.name) LIKE '%emma%')
- Never use exact equality for customer names
- customers.id is the primary key; use customers.id not customer_id when counting customers
- Only use SELECT statements
- When listing all customers, include all columns: id, name, email, plan, joined_date
{ticket_hint}
Schema:
{schema}

Question: {question}

SQL:"""
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def run_sql(sql: str):
    if _SQL_BLOCKLIST.search(sql):
        raise ValueError("Query contains disallowed SQL operation.")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
    return columns, rows


def format_results(columns: list, rows: list, question: str) -> str:
    if not rows:
        return (
            "No results found for your query. "
            "If you were searching by name, try a different spelling "
            "(e.g. \"Find customer Emma\" instead of \"Ema\")."
        )

    rows_text = "\n".join(
        "  " + ", ".join(f"{col}: {val}" for col, val in zip(columns, row))
        for row in rows
    )

    # Detect if the result contains ticket columns from a JOIN
    has_tickets = any(c in columns for c in ("ticket_id", "subject", "status", "ticket_date"))
    ticket_instruction = (
        "\nThe data includes ticket rows joined to a customer profile. "
        "Show the customer profile details ONCE, then list ALL tickets as a numbered list "
        "with subject, status, and date. If ticket_id is NULL the customer has no tickets."
        if has_tickets else ""
    )

    prompt = f"""You are a helpful customer support assistant. Format this database result as a clear, friendly response.

Original question: {question}
{ticket_instruction}
Data:
{rows_text}

Provide a concise, helpful answer:"""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _shorten_like_patterns(sql: str) -> str:
    """Shorten LIKE '%xyz%' patterns by 1 char to handle partial name typos."""
    def shorten(m):
        inner = m.group(1)
        if len(inner) > 2:
            return f"LIKE '%{inner[:-1]}%'"
        return m.group(0)
    return re.sub(r"LIKE\s+'%([^%]+)%'", shorten, sql, flags=re.IGNORECASE)


def query_database(question: str, original_query: str = "") -> str:
    try:
        schema = get_schema()
        # Use original user query for ticket detection — the MCP sub_query may
        # be a rephrased version that drops keywords like "ticket" or "history".
        detection_question = original_query if original_query else question
        sql = nl_to_sql(question, schema, wants_tickets=_wants_tickets(detection_question))
        sql = sql.replace("```sql", "").replace("```", "").strip()
        columns, rows = run_sql(sql)

        # Progressively shorten LIKE patterns to handle typos (e.g. Ema → Emma)
        if not rows and "LIKE" in sql.upper():
            for _ in range(3):
                sql = _shorten_like_patterns(sql)
                columns, rows = run_sql(sql)
                if rows:
                    break

        return format_results(columns, rows, question)
    except ValueError as e:
        logger.warning("Blocked query: %s — %s", question[:80], e)
        return "Sorry, that type of query is not allowed."
    except anthropic.AuthenticationError:
        logger.error("Anthropic API key is invalid or expired")
        return "⚠️ API key is invalid or expired. Please update ANTHROPIC_API_KEY in your .env file."
    except Exception as e:
        logger.error("SQL agent error for question '%s': %s", question[:80], e, exc_info=True)
        return "Sorry, I couldn't retrieve that information. Please try rephrasing your question."

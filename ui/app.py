import sys
import os
import logging
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import streamlit as st
from agents.rag_agent import index_pdfs, index_single_pdf, get_indexed_filenames, get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POLICIES_DIR = os.getenv("POLICIES_DIR", "data/pdfs")
DB_PATH = os.getenv("DB_PATH", "data/support.db")
MCP_URL = os.getenv("MCP_URL", "")          # e.g. http://mcp:8000 in Docker
MAX_PDF_BYTES = 10 * 1024 * 1024            # 10 MB

# Module-level flag — persists between Streamlit reruns in the same process.
# Ensures we only spawn one background thread, not one per user interaction.
_PREWARM_STARTED = False


def _maybe_prewarm():
    global _PREWARM_STARTED
    if _PREWARM_STARTED:
        return
    _PREWARM_STARTED = True

    def _load():
        try:
            get_embeddings()
            logger.info("Embedding model pre-warmed and ready for RAG queries.")
        except Exception as exc:
            logger.warning("Pre-warm failed (non-fatal): %s", exc)

    threading.Thread(target=_load, daemon=True).start()


_maybe_prewarm()


def _ask_via_mcp(prompt: str) -> dict:
    """POST to MCP server. Falls back to direct graph call if MCP_URL is unset or unreachable."""
    if MCP_URL:
        try:
            import httpx
            r = httpx.post(
                f"{MCP_URL}/query",
                json={"message": prompt},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            return {"output": data["response"], "route": data["agent"]}
        except Exception as e:
            logger.warning("MCP server call failed (%s), falling back to direct call", e)

    # Direct call (local dev or MCP unreachable)
    from graph.router import ask
    return ask(prompt)


# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="Support Hub — AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 16px;
    padding: 48px 32px 40px;
    text-align: center;
    margin-bottom: 28px;
}
.hero h1 { color: #fff; font-size: 2.2rem; font-weight: 700; margin: 0 0 10px; }
.hero p  { color: #c7d2fe; font-size: 1.05rem; margin: 0; }
.badge-sql {
    display: inline-block;
    background: #dbeafe; color: #1d4ed8;
    font-size: .75rem; font-weight: 600;
    padding: 2px 10px; border-radius: 999px;
    margin-bottom: 6px;
}
.badge-rag {
    display: inline-block;
    background: #d1fae5; color: #065f46;
    font-size: .75rem; font-weight: 600;
    padding: 2px 10px; border-radius: 999px;
    margin-bottom: 6px;
}
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prefill" not in st.session_state:
    st.session_state.prefill = None
if "show_home" not in st.session_state:
    st.session_state.show_home = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Support Hub")
    st.caption("Powered by Claude Haiku · MCP Tool Use")
    st.divider()

    # Live DB metrics
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM customers")
            n_customers = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM tickets WHERE status='open'")
            n_open = cur.fetchone()[0]
        c1, c2 = st.columns(2)
        c1.metric("Customers", n_customers)
        c2.metric("Open Tickets", n_open)
    except Exception:
        pass

    # Back to home — only shown during an active conversation
    if st.session_state.messages:
        st.markdown("")
        if st.button("🏠 New conversation", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.prefill = None
            st.rerun()

    st.divider()
    st.markdown("### 📄 Policy Documents")
    st.caption("Index PDFs to enable refund, cancellation & service agreement answers.")

    # Read indexed filenames from ChromaDB (no model load needed)
    indexed = get_indexed_filenames()
    if indexed:
        for fname in indexed:
            st.markdown(f"✅ {fname}")
    else:
        st.caption("No documents indexed yet — click 'Index all PDFs' below.")

    if st.button("🔄 Index all PDFs", use_container_width=True):
        try:
            with st.spinner("Indexing PDFs… loading embedding model on first run (~15s)."):
                added = index_pdfs(POLICIES_DIR)
            if added:
                st.success(f"Indexed {added} new chunks.")
                st.rerun()
            else:
                st.info("All PDFs already indexed.")
        except Exception as e:
            logger.error("Index all error: %s", e, exc_info=True)
            st.error(f"Indexing failed: {e}")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        if uploaded.size > MAX_PDF_BYTES:
            st.error(f"File too large ({uploaded.size // 1024 // 1024} MB). Max 10 MB.")
        elif uploaded.name in indexed:
            st.info(f"✅ **{uploaded.name}** is already indexed. Ready to query!")
        else:
            save_path = os.path.join(POLICIES_DIR, uploaded.name)
            os.makedirs(POLICIES_DIR, exist_ok=True)
            try:
                with open(save_path, "wb") as f:
                    f.write(uploaded.read())
                with st.spinner(f"Indexing {uploaded.name}… loading embedding model on first run (~15s)."):
                    chunks = index_single_pdf(save_path)
                st.success(f"Indexed **{chunks}** chunks from {uploaded.name}.")
                st.rerun()
            except Exception as e:
                logger.error("PDF upload error: %s", e, exc_info=True)
                st.error(f"Failed to index PDF: {e}")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Hi Froila, how can I help?</h1>
  <p>Ask about customers, tickets, billing, or company policies.</p>
</div>
""", unsafe_allow_html=True)

# ── Back to home button (shown during active conversation) ────────────────────
if st.session_state.messages:
    if st.button("← Back to topics", key="back_main"):
        st.session_state.messages = []
        st.session_state.prefill = None
        st.rerun()
    st.divider()

# ── Category cards (shown before first message) ───────────────────────────────
# Queries are generic so the MCP tool + LLM handle them dynamically.
CARDS = [
    ("👤", "Customer Profiles",  "Look up any customer",       "Show me all customer profiles"),
    ("🎫", "Support Tickets",    "View open or past tickets",   "List all open support tickets"),
    ("💳", "Billing & Plans",    "Plans, payments, upgrades",   "Show a breakdown of customers by subscription plan"),
    ("💰", "Refund Policy",      "Understand refund rules",     "What is the current refund policy?"),
    ("📋", "Cancellation Terms", "Cancellation & termination",  "What are the cancellation terms?"),
    ("📄", "Service Agreement",  "Full policy document",        "Summarise the key points of the service agreement"),
]

if not st.session_state.messages:
    cols = st.columns(3)
    for i, (icon, title, desc, query) in enumerate(CARDS):
        with cols[i % 3]:
            if st.button(f"{icon} **{title}**\n\n{desc}", key=f"card_{i}", use_container_width=True):
                st.session_state.prefill = query
                st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("agent"):
            badge = "badge-sql" if msg["agent"] == "sql" else "badge-rag"
            label = "🗄 SQL Agent" if msg["agent"] == "sql" else "📚 RAG Agent"
            st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
prefill_value = st.session_state.prefill or ""
st.session_state.prefill = None

prompt = st.chat_input("Ask about a customer, ticket, or company policy…") or prefill_value

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Detect whether this is likely a policy (RAG) query so we can show a
        # better spinner message if the embedding model hasn't warmed up yet.
        _rag_keywords = {"refund", "cancel", "cancellation", "policy", "agreement",
                         "terms", "service", "eligible", "clause"}
        _is_rag = any(w in prompt.lower() for w in _rag_keywords)
        _spinner_msg = (
            "Searching policy documents… (first search loads the model, ~15–30s)"
            if _is_rag else "Thinking…"
        )
        with st.spinner(_spinner_msg):
            try:
                result = _ask_via_mcp(prompt)
                response = result["output"]
                route = result["route"]
            except Exception as e:
                logger.error("_ask_via_mcp() failed: %s", e, exc_info=True)
                response = "Sorry, something went wrong. Please try again."
                route = "sql"

        badge = "badge-sql" if route == "sql" else "badge-rag"
        label = "🗄 SQL Agent" if route == "sql" else "📚 RAG Agent"
        st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response, "agent": route})

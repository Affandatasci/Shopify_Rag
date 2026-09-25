"""
Norhaven & Co. RAG demo -- agent.

Two tools, one LangGraph ReAct agent (langgraph.prebuilt.create_react_agent):
  - search_policies(query)     -> semantic search over the ingested PDF (Qdrant)
  - lookup_order(order_number) -> mock "live" order-status check (orders.json)

The model decides which tool(s) a question needs -- that's what makes this
an agent instead of a plain "read the FAQ" bot. "Where's my order #1004"
never touches the PDF at all; it goes straight to lookup_order. A question
like "I bought this on sale, can I return it after 20 days" forces
search_policies to surface the full-price vs Sale vs Clearance distinction
rather than a single keyword match.

app.py imports answer() from here -- nothing in this file is meant to be
run directly.
"""

import json
import os

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

QDRANT_URL = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION = os.environ.get("QDRANT_COLLECTION", "norhaven_demo")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MAIN_MODEL = os.environ.get("MAIN_MODEL", "openai/gpt-oss-120b")

# Swapped from mxbai-embed-large-v1 (335M params, ~1.3GB) to bge-small
# (33M params, ~130MB) to fit Streamlit Community Cloud's free-tier RAM --
# the large model was pushing the app's memory over the limit and causing
# repeated crash-restarts. bge-small uses the same "prefix the query"
# convention, so search_policies() below didn't need to change at all.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
ORDERS_PATH = os.path.join(os.path.dirname(__file__), "orders.json")
TOP_K = 4

HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

# Loaded once at import time, forced to CPU -- same fix used on Forge
# Physique: letting sentence-transformers auto-detect CUDA on a shared /
# virtual GPU produced NaN vectors there. This demo has no GPU at all, so
# CPU is not a workaround here, it's just the only device.
_embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

with open(ORDERS_PATH) as f:
    _ORDERS = {o["order_number"]: o for o in json.load(f)}


@tool
def search_policies(query: str) -> str:
    """Search Norhaven & Co.'s policies, product catalog, sizing guide, and
    FAQ for information relevant to the query. Use this for any question
    about products, sizing, shipping, returns, discounts, loyalty, store
    locations, or company policy. Always use this before answering a policy
    or product question -- never answer one from memory."""
    # bge-small-en-v1.5's model card asks for this prefix on queries
    # only -- the documents embedded in ingest.py are NOT prefixed.
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vector = _embed_model.encode(prefixed, normalize_embeddings=True).tolist()

    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        headers=HEADERS,
        json={"vector": vector, "limit": TOP_K, "with_payload": True},
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json()["result"]
    if not hits:
        return "No matching content found in Norhaven's policies."

    blocks = [f"[page {h['payload']['page']}] {h['payload']['text']}" for h in hits]
    return "\n\n".join(blocks)


@tool
def lookup_order(order_number: str) -> str:
    """Look up the live status of a Norhaven & Co. order by its order
    number, e.g. "1004". Use this whenever a customer asks about the
    status, tracking, or delivery of a specific order. Never guess an
    order's status from search_policies -- policy text never contains real
    order data, only lookup_order does."""
    order = _ORDERS.get(order_number.strip())
    if not order:
        return (
            f"No order found with number {order_number}. Ask the customer "
            "to double-check the number in their confirmation email."
        )
    return json.dumps(order, indent=2)


SYSTEM_PROMPT = """You are the customer support assistant for Norhaven & Co., \
an apparel and home-goods company. Answer only from what search_policies and \
lookup_order return -- never invent a policy, price, or order status.

When a return or exchange question depends on how an item was priced (full \
price, Sale, or Clearance), check that distinction explicitly before \
answering, since the three have different rules. If a question needs both \
tools (for example "can I return my order, it's #1004"), call both before \
answering. Keep answers short and direct, in plain customer-support \
language. If nothing relevant is found, say so instead of guessing."""

_model = ChatGroq(model=MAIN_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
_graph = create_react_agent(_model, tools=[search_policies, lookup_order], prompt=SYSTEM_PROMPT)


def answer(message: str, history: list) -> str:
    """history is Gradio's list of {"role": "user"/"assistant", "content": str} dicts."""
    messages = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    result = _graph.invoke({"messages": messages})

    for msg in reversed(result["messages"]):
        content = getattr(msg, "content", "")
        if isinstance(msg, AIMessage) and content:
            return content
    return "Sorry, I couldn't come up with an answer to that -- please try rephrasing."

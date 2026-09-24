# Norhaven & Co. — Store Assistant

A RAG demo chat agent for **Norhaven & Co.**, a fictional apparel and home-goods Shopify store. Ask it about sizing, shipping, returns, store policy, or the status of a specific order.

Built the same way as the Forge Physique demo: a two-tool [LangGraph](https://langchain-ai.github.io/langgraph/) ReAct agent, running on [Groq](https://groq.com) for inference and [Qdrant Cloud](https://qdrant.tech) for retrieval, served through a [Streamlit](https://streamlit.io) chat UI.

## How it works

The agent (`agent.py`) is a `langgraph.prebuilt.create_react_agent` with two tools, and the model decides which one(s) a question needs:

| Tool | Purpose | Source |
|---|---|---|
| `search_policies(query)` | Semantic search over sizing, shipping, returns, discounts, loyalty, and store-policy text | `norhaven_policies_faq.pdf`, embedded into Qdrant |
| `lookup_order(order_number)` | Live order status, tracking, and delivery lookup | `orders.json` (mock order data, #1001–1010) |

A question like *"Where's my order #1004?"* never touches the PDF — it goes straight to `lookup_order`. A question like *"I bought this on sale, can I return it after 20 days?"* forces `search_policies` to surface the full-price vs. Sale vs. Clearance distinction rather than matching on a single keyword. A question that needs both ("can I return my order, it's #1004") calls both tools before answering.

## Files

- `app.py` — Streamlit chat UI
- `agent.py` — the LangGraph agent and its two tools
- `ingest.py` — one-time script: extracts, chunks, embeds, and upserts `norhaven_policies_faq.pdf` into Qdrant
- `generate_pdf.py` — dev-only script that built `norhaven_policies_faq.pdf` (reportlab); not needed to run the demo
- `orders.json` — mock order data backing `lookup_order`
- `norhaven_policies_faq.pdf` — the store's policy/product/FAQ document

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
python ingest.py       # one-time: embeds the PDF into Qdrant
streamlit run app.py   # starts the chat UI
```

`ingest.py` can also be run as a Colab notebook instead of locally — same extraction/chunking/embedding logic, just no local `torch`/`sentence-transformers` install needed. Either route fully populates the Qdrant collection; running both just re-does (and wipes, since ingestion resets the collection) the same work.

### Environment variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key, used by `agent.py` for the chat model |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | Collection name (default `norhaven_demo`) — use a distinct collection from any other demo sharing the same cluster |
| `MAIN_MODEL` | Groq model id (default `openai/gpt-oss-120b`) |

See `.env.example`. Never commit a real `.env` file — `.gitignore` already excludes it.

## Deploy

Deployed on **Streamlit Community Cloud** (free, no card required). It deploys straight from a GitHub repo:

1. `git init`, commit, push this folder to a new GitHub repo (public or private — either works for a free deploy).
2. On [share.streamlit.io](https://share.streamlit.io), sign in with GitHub → **New app** → pick the repo/branch → set **Main file path** to `app.py`.
3. Before (or right after) clicking Deploy, open **Advanced settings → Secrets** and paste in `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` as **root-level** keys (plain `KEY = "value"` lines, not nested under a `[section]`) — Streamlit copies root-level secrets into `os.environ` automatically, which is what `agent.py` reads from. Never commit real key values to the repo.
4. Deploy. First boot takes a few minutes (installing `torch`/`sentence-transformers`); after that you get a public `https://<something>.streamlit.app` link anyone can open.

Run `ingest.py` (locally or via Colab) against the same Qdrant collection before or after the app goes live — the deployed app itself only ever reads from Qdrant, it never ingests.

**Free-tier limits to know:** the app sleeps after ~12 hours with no visitors and takes 30s–1min to wake up on the next visit (a "app is waking up" screen, not a failure). RAM is roughly 2.7GB, which comfortably fits this app's footprint. There's a soft cap on how many free apps one account can run at once — if you hit it, you can stop/delete an older app to free a slot.

## Notes

- **CPU-forced embeddings**: `SentenceTransformer(..., device="cpu")` is set explicitly in both `agent.py` and `ingest.py`. Streamlit Community Cloud's free tier is CPU-only anyway, but this is kept explicit because auto-detecting CUDA on a shared/virtual GPU produced NaN vectors on Forge Physique — worth keeping even if you ever move this to GPU-backed hosting.
- **Qdrant over plain REST, not `qdrant-client`**: both `ingest.py` and `agent.py` call Qdrant Cloud's REST API directly with `requests` rather than the `qdrant-client` library, since `search()`/`query_points()` broke across client versions on Forge Physique. Keeping to plain HTTP means there's only one thing that has to keep working.
- **Embed prefix on queries only**: `mxbai-embed-large-v1`'s model card asks for the `"Represent this sentence for searching relevant passages: "` prefix on search queries. It's applied in `search_policies` at query time and must **never** be applied to documents at ingest time (`ingest.py` embeds raw chunk text).
- **Chat history format**: `agent.py`'s `answer()` builds and expects chat history as OpenAI-style `{"role", "content"}` dicts. `app.py` keeps `st.session_state.messages` in exactly that shape, so no conversion is needed anywhere.
- **`st.session_state` instead of a callback's return value**: Streamlit reruns the whole script top-to-bottom on every interaction and has no built-in chat-history object, so `app.py` keeps the conversation in `st.session_state.messages` itself (initialized once, appended to on each turn) rather than passing history around as a function argument/return value the way the old Gradio `respond()` did.
- **CSS targets Streamlit's `data-testid` hooks, not generated class names**: Streamlit's own CSS classes (`st-emotion-cache-...`) are content-hashed and change between versions/reruns, so they're not safe to style against. `app.py`'s CSS instead targets stable `data-testid` attributes (`stChatMessage`, `stChatMessageAvatarUser`/`Assistant`, `stChatInput`, etc.), including a `:has()` selector to color user vs. assistant bubbles differently based on which avatar each message contains — verified against the actual rendered DOM before shipping.

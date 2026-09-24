"""
Norhaven & Co. RAG demo -- Streamlit UI.

Setup:
    1. pip install -r requirements.txt
    2. Copy .env.example to .env and fill in GROQ_API_KEY, QDRANT_URL,
       QDRANT_API_KEY -- reuse the same Groq and Qdrant Cloud accounts as
       Forge Physique, just set QDRANT_COLLECTION to a new collection name.
    3. python ingest.py            (one-time: embeds the PDF into Qdrant)
    4. streamlit run app.py        (starts the chat UI)

Deploy on Streamlit Community Cloud (free, no card required) -- see
README.md for the full step-by-step. Short version: push this repo to
GitHub, connect it at share.streamlit.io, set app.py as the entrypoint,
and add GROQ_API_KEY / QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION as
root-level (not nested under a table) keys in the app's Secrets settings --
Streamlit Community Cloud copies root-level secrets into os.environ
automatically, so agent.py's os.environ[...] calls work unchanged. Never
commit real key values to this repo.
"""

import streamlit as st

from agent import answer

TITLE = "Norhaven & Co. — Store Assistant"
DESCRIPTION = (
    "Demo RAG assistant for a fictional Shopify store. Ask about sizing, "
    "shipping, returns, or store policy — or ask about a specific "
    "order (try order numbers 1001–1010)."
)

# Palette pulled from the storefront mark: maroon -> red -> orange -> yellow -> blue
MAROON = "#7A1E1E"
RED = "#D62828"
ORANGE = "#F4900C"
YELLOW = "#F4B400"
BLUE = "#1B5FA8"
BLUE_LIGHT = "#D9EAFB"
NAVY_INK = "#0F3D66"

st.set_page_config(page_title=TITLE, page_icon="🛍️", layout="centered")

# Larger fonts throughout (for recording/social use) + the brand palette on
# the title banner, chat bubbles, and buttons. Selectors below use
# Streamlit's own data-testid hooks (verified against the real rendered
# DOM), not generated class names, so they keep working across reruns.
CUSTOM_CSS = f"""
<style>
html, body, [class*="css"] {{
    font-size: 19px !important;
}}

.nh-banner {{
    background: linear-gradient(90deg, {MAROON}, {RED}, {ORANGE}, {YELLOW}, {BLUE});
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 20px;
}}
.nh-banner h1 {{
    font-size: 42px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 0 10px 0 !important;
}}
.nh-banner p {{
    font-size: 22px !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    opacity: 0.95;
    margin: 0 !important;
}}

[data-testid="stChatMessage"] {{
    font-size: 20px !important;
    border-radius: 14px !important;
}}
[data-testid="stChatMessage"] p {{
    font-size: 20px !important;
    line-height: 1.5 !important;
}}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    background: {BLUE_LIGHT} !important;
    border: 1px solid {BLUE} !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p {{
    color: {NAVY_INK} !important;
}}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
    background: #FFF6E0 !important;
    border: 1px solid {YELLOW} !important;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] p {{
    color: {MAROON} !important;
}}

[data-testid="stChatInput"] textarea {{
    font-size: 20px !important;
}}

.stButton > button {{
    background: {YELLOW} !important;
    color: {MAROON} !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    border-radius: 10px !important;
    border: none !important;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    f'<div class="nh-banner"><h1>{TITLE}</h1><p>{DESCRIPTION}</p></div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about sizing, returns, shipping, or an order number...")
if prompt:
    history_snapshot = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = answer(prompt, history_snapshot)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

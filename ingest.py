"""
Norhaven & Co. RAG demo -- one-time ingestion script.

Reads norhaven_policies_faq.pdf, splits it into chunks, embeds each chunk
with mxbai-embed-large-v1 (forced to CPU -- same fix used on Forge Physique
to avoid ZeroGPU virtual-CUDA producing NaN vectors), and upserts everything
into Qdrant Cloud over the plain REST API.

Deliberately not using qdrant-client's search()/query_points() here: on
Forge Physique, search() was removed in qdrant-client v1.16.0 and
query_points() was rejected by the server, so the working fix was to drop
the client library for queries and call Qdrant's REST endpoint directly
with `requests`. This script does the same for both writing and reading,
so there's only one thing (plain HTTP) that has to work.

Run this once, before app.py is used:
    python ingest.py

Requires QDRANT_URL and QDRANT_API_KEY (see .env.example) -- reuse the same
Qdrant Cloud account/cluster as Forge Physique, just point
QDRANT_COLLECTION at a new collection so the two demos don't share data.
"""

import os
import sys
import uuid

import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = os.path.join(os.path.dirname(__file__), "norhaven_policies_faq.pdf")
QDRANT_URL = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION = os.environ.get("QDRANT_COLLECTION", "norhaven_demo")
EMBED_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
EMBED_DIM = 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}


def extract_text(pdf_path):
    """Return [{"page": n, "text": "..."}] for every page with real text."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages):
    """Split each page's text into overlapping chunks, tagged with the page
    it came from so answers can point back to a real page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for p in pages:
        for piece in splitter.split_text(p["text"]):
            piece = piece.strip()
            if piece:
                chunks.append({"text": piece, "page": p["page"]})
    return chunks


def reset_collection():
    """Wipe and recreate the collection so re-running ingest.py is safe."""
    requests.delete(f"{QDRANT_URL}/collections/{COLLECTION}", headers=HEADERS, timeout=30)
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers=HEADERS,
        json={"vectors": {"size": EMBED_DIM, "distance": "Cosine"}},
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Failed to create collection: {resp.status_code} {resp.text}")
    print(f"Collection '{COLLECTION}' created fresh.")


def upsert_chunks(chunks, embeddings):
    points = []
    for chunk, vector in zip(chunks, embeddings):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector.tolist(),
            "payload": {
                "text": chunk["text"],
                "page": chunk["page"],
                "source": "Norhaven & Co. - Policies, Product Guide & Customer FAQ",
            },
        })
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
        headers=HEADERS,
        json={"points": points},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to upsert points: {resp.status_code} {resp.text}")
    print(f"Upserted {len(points)} chunks.")


def main():
    if not os.path.exists(PDF_PATH):
        sys.exit(f"Can't find {PDF_PATH} -- put norhaven_policies_faq.pdf next to ingest.py first.")

    print("Reading PDF...")
    pages = extract_text(PDF_PATH)
    print(f"Extracted text from {len(pages)} pages.")

    print("Chunking...")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading embedding model ({EMBED_MODEL_NAME}) on CPU...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    print("Embedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)

    print("Resetting Qdrant collection...")
    reset_collection()

    print("Upserting into Qdrant...")
    upsert_chunks(chunks, embeddings)

    print("Done -- agent.py can now query the collection.")


if __name__ == "__main__":
    main()

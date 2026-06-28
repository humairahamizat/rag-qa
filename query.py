"""
query.py - Step 2: Ask questions over your document corpus.
Usage:
    python query.py                              # interactive mode
    python query.py "What is AI governance?"     # single question
    python query.py --examples                   # run 3 built-in examples
"""

import sys
import pickle
import numpy as np
import faiss
import os
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_FILE  = "index.faiss"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5          # how many chunks to retrieve per query
MAX_TOKENS  = 800        # max length of the generated answer
# ──────────────────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "What are the main risks of AI identified in these documents?",
    "How do different countries or organisations approach AI regulation?",
    "What role does transparency play in AI governance?",
]


def load_resources():
    """Load the FAISS index, chunk metadata, and embedding model."""
    print("Loading index and model (first call may take ~30 s) ...")
    index  = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    print(f"Ready — {index.ntotal} vectors indexed across {len(set(c['doc_id'] for c in chunks))} documents.\n")
    return index, chunks, model


def retrieve(query: str, index, chunks: list[dict], model, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and find the top-k most similar chunks."""
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk = dict(chunks[idx])          # copy so we don't mutate
        chunk["score"] = float(dist)
        results.append(chunk)
    return results


def build_prompt(query: str, retrieved: list[dict]) -> str:
    """Combine retrieved chunks into a prompt for the LLM."""
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        context_parts.append(f"[Source {i} — Document {r['doc_id']}]\n{r['text']}")
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are an expert assistant specialising in AI governance and policy.
Answer the user's question using ONLY the context provided below.
- Be specific and cite which document (by its ID) your answer comes from.
- If the context does not contain enough information, say so clearly.
- Do not make up facts not present in the context.

=== CONTEXT ===
{context}

=== QUESTION ===
{query}

=== ANSWER ==="""


def answer_query(query: str, index, chunks: list[dict], model) -> str:
    """Retrieve relevant chunks and generate a grounded answer."""
    retrieved = retrieve(query, index, chunks, model)

    # Show which docs were retrieved (helpful for debugging)
    doc_ids = [r["doc_id"] for r in retrieved]
    print(f"  Retrieved from docs: {doc_ids}")

    prompt = build_prompt(query, retrieved)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def run_examples(index, chunks, model):
    """Run the built-in example queries and print results."""
    print("=" * 70)
    print("RUNNING EXAMPLE QUERIES")
    print("=" * 70)
    for q in EXAMPLE_QUERIES:
        print(f"\n Question: {q}")
        print("-" * 60)
        ans = answer_query(q, index, chunks, model)
        print(ans)
        print("=" * 70)


def interactive_mode(index, chunks, model):
    """Keep asking for questions until the user types 'exit'."""
    print("Interactive mode — type your question and press Enter.")
    print("Type 'exit' or press Ctrl+C to quit.\n")
    while True:
        try:
            query = input(" Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        print()
        ans = answer_query(query, index, chunks, model)
        print(f"\n Answer:\n{ans}\n")
        print("-" * 60)


def main():
    # Check for the index files before doing anything else
    import os
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        print("ERROR: Index not found. Run  python ingest.py  first.")
        sys.exit(1)

    import os

    index, chunks, model = load_resources()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--examples":
            run_examples(index, chunks, model)
        else:
            # Single question passed as command-line argument
            query = " ".join(sys.argv[1:])
            print(f" Question: {query}\n")
            ans = answer_query(query, index, chunks, model)
            print(f" Answer:\n{ans}")
    else:
        interactive_mode(index, chunks, model)


if __name__ == "__main__":
    main()

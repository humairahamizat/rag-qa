"""
ingest.py - Step 1: Read all documents, chunk them, embed them, save FAISS index.
Run this ONCE before using query.py.
"""

import os
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR   = "data/agora/fulltext"   # folder with all .txt files
CHUNK_SIZE = 500                      # words per chunk
OVERLAP    = 50                       # overlap between chunks (helps context)
INDEX_FILE = "index.faiss"            # where the vector index is saved
CHUNKS_FILE = "chunks.pkl"            # where chunk metadata is saved
EMBED_MODEL = "all-MiniLM-L6-v2"     # fast + good quality embeddings (~80MB)
# ──────────────────────────────────────────────────────────────────────────────


def chunk_text(text: str, doc_id: str) -> list[dict]:
    """Split a document into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - OVERLAP
    for i in range(0, len(words), step):
        chunk_words = words[i : i + CHUNK_SIZE]
        chunk_text  = " ".join(chunk_words).strip()
        if len(chunk_words) < 20:          # skip tiny trailing fragments
            continue
        chunks.append({
            "doc_id": doc_id,
            "text":   chunk_text,
            "start":  i,
        })
    return chunks


def load_documents(docs_dir: str) -> list[dict]:
    """Read every .txt file in docs_dir and return a list of chunks."""
    all_chunks = []
    files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]
    print(f"Found {len(files)} documents in {docs_dir}")

    for fname in tqdm(files, desc="Reading & chunking docs"):
        doc_id = fname.replace(".txt", "")
        path   = os.path.join(docs_dir, fname)
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            all_chunks.extend(chunk_text(text, doc_id))
        except Exception as e:
            print(f"  ⚠  Skipping {fname}: {e}")

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Encode all chunk texts into vectors using SentenceTransformer."""
    print(f"\nLoading embedding model '{EMBED_MODEL}' ...")
    model  = SentenceTransformer(EMBED_MODEL)
    texts  = [c["text"] for c in chunks]
    print("Embedding chunks — this takes 3-5 minutes on first run ...")
    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def build_and_save_index(embeddings: np.ndarray, chunks: list[dict]) -> None:
    """Build a FAISS flat L2 index and save both index + chunk metadata."""
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)   # exact nearest-neighbour search
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"✅  FAISS index saved → {INDEX_FILE}  ({index.ntotal} vectors)")

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅  Chunk metadata saved → {CHUNKS_FILE}")


def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"ERROR: Documents folder not found at '{DOCS_DIR}'")
        print("Make sure you extracted the zip so the path looks like:")
        print("  data/agora/fulltext/1.txt  etc.")
        return

    chunks     = load_documents(DOCS_DIR)
    embeddings = embed_chunks(chunks)
    build_and_save_index(embeddings, chunks)

    print("\n🎉  Ingestion complete!  Now run:  python query.py")


if __name__ == "__main__":
    main()

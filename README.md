Download dataset from: https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data

# AI Governance RAG Q&A System

A retrieval-augmented generation (RAG) system that answers questions over a corpus of AI governance documents.

## How It Works

```
User Question
     │
     ▼
[Embed query]  ──►  SentenceTransformer (all-MiniLM-L6-v2)
     │
     ▼
[Vector search]  ──►  FAISS index (top 5 most similar chunks)
     │
     ▼
[Generate answer]  ──►  Groq llama-3.3-70b-versatile (grounded, citing doc IDs)
     │
     ▼
Answer + Sources
```

**Ingestion pipeline (`ingest.py`):**
1. Reads all `.txt` documents from `data/agora/fulltext/`
2. Splits each document into 500-word chunks with 50-word overlap
3. Embeds every chunk using `all-MiniLM-L6-v2` (a fast, high-quality sentence embedding model)
4. Stores all vectors in a FAISS flat index for exact nearest-neighbour search

**Query pipeline (`query.py`):**
1. Embeds the user's question with the same model
2. Retrieves the 5 most similar chunks from FAISS
3. Sends those chunks + the question to Groq with a strict grounding prompt
4. Returns the answer with citations to source document IDs

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Fast, small (~80MB), excellent quality for semantic search |
| Chunk size | 500 words, 50-word overlap | Balances context richness with retrieval precision |
| Vector index | FAISS FlatL2 | Exact search, no tuning needed, fast enough for this corpus size |
| LLM | Groq llama-3.3-70b-versatile | Free tier, fast inference, strong instruction-following |
| Top-k | 5 chunks | Enough context without overloading the prompt |

## Assumptions Made

- Documents are in plain `.txt` format under `data/agora/fulltext/`
- Questions are in English (embedding model optimised for English)
- Groq free tier is sufficient for evaluation purposes
- 500-word chunks are appropriate for this document type
- Top 5 retrieved chunks provide enough context for accurate answers

## Limitations

- **No reranking:** Retrieved chunks are ranked purely by embedding similarity. A cross-encoder reranker would improve precision.
- **Fixed chunk size:** Some documents are very short; others are very long. Adaptive chunking (e.g., by paragraph or sentence) could improve quality.
- **No metadata filtering:** The system doesn't filter by document date, author, or category — adding this would allow more targeted queries.
- **Single-turn only:** The CLI doesn't maintain conversation history. Multi-turn support would require storing previous exchanges.

## How to Run

### Prerequisites

- Python 3.10+
- A Groq API key (get one free at console.groq.com)

### 1. Install dependencies

```bash
pip install sentence-transformers faiss-cpu groq tqdm
```

### 2. Set up the data

Download the dataset from: https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data

Extract the corpus zip file so the structure looks like:
```
data/
  agora/
    fulltext/
      1.txt
      2.txt
      ...
```

### 3. Build the index (run once)

```bash
python ingest.py
```

This creates `index.faiss` and `chunks.pkl`. Takes ~3-5 minutes.

### 4. Set your API key

```bash
# Windows
set GROQ_API_KEY=gsk_...

# Mac/Linux
export GROQ_API_KEY=gsk_...
```

### 5. Ask questions

```bash
# Interactive mode
python query.py

# Single question
python query.py "What are the main risks of AI?"

# Run built-in examples
python query.py --examples
```

## Example Queries & Responses

### Q: What are the main risks of AI identified in these documents?

> The main risks identified include algorithmic bias and discrimination, lack of transparency in automated decision-making, risks to privacy from large-scale data collection, and potential misuse in surveillance contexts (Documents 1032, 1044, 1026).

### Q: How do different countries approach AI regulation?

> Different countries take varying approaches - some favour risk-based frameworks classifying AI systems by risk level, while others adopt principles-based sector-led approaches. International organisations like the OECD provide non-binding principles centred on transparency, accountability, and human-centred values (Documents 432, 757).

### Q: What role does transparency play in AI governance?

> Transparency is identified as a foundational principle across multiple documents. It encompasses explainability of model decisions, disclosure of training data sources, and clear communication of AI system capabilities and limitations to end users.

## What I Would Do Next (Given More Time)

1. **Add a reranker** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve retrieval precision
2. **Metadata-aware retrieval** using the `documents.csv` and `authorities.csv` files to filter by source, date, or country
3. **Evaluation harness** - generate a test set of question/answer pairs and measure retrieval recall and answer faithfulness
4. **Hybrid search** - combine BM25 keyword search with vector search for better coverage of exact terms
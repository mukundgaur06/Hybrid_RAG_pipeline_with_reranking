# HybridRAG: Document QA with Hybrid Retrieval & Cross-Encoder Reranking

HybridRAG is an educational, production-oriented Retrieval-Augmented Generation (RAG) system built from first principles in Python. It pairs **lexical search (BM25)** and **dense vector search (Bi-Encoder)** using **Reciprocal Rank Fusion (RRF)**, followed by a **Cross-Encoder reranker** for high-precision document question-answering.

---

## Architecture at a Glance

```
Raw Documents (PDF, TXT, MD) ──► Chunker (with metadata)
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
         Lexical Index (BM25)                      Dense Index (Bi-Encoder)
                 │                                           │
                 └──────────────┬────────────────────────────┘
                                ▼
                   Reciprocal Rank Fusion (RRF)
                                ▼
                   Cross-Encoder Reranker
                 (ms-marco-MiniLM-L6-v2)
                                │
                 ┌──────────────┴────────────────────────────┐
                 ▼                                           ▼
       Grounded Generation                      Evidence Viewer Mode
       (Optional Gemini API)                     (Zero-API-key fallback)
```

For complete technical specifications, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
For core concept definitions (chunking, embeddings, RRF, Cross-Encoder, MRR), see [`docs/GLOSSARY.md`](docs/GLOSSARY.md).
For the phase-by-phase roadmap, see [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Project Structure

```text
Hybrid_RAG_pipeline_with_reranking/
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions (caches, venvs, weights, keys)
├── README.md                 # Project overview and setup guide
├── pyproject.toml            # Python package metadata and tool configs
├── requirements.txt          # Pinned dependency requirements
├── docs/
│   ├── ARCHITECTURE.md       # Detailed technical design & diagrams
│   ├── GLOSSARY.md           # Beginner-friendly ML engineering concepts
│   └── ROADMAP.md            # Phase-by-phase execution plan
├── evaluation/
│   ├── README.md             # Benchmark description and metrics guide
│   └── benchmark_dataset.json# Evaluation queries, targets & ground truths
├── examples/                 # Executable scripts and demos
│   └── demo_retrieval.py     # Live demonstration of Lexical, Dense, and RRF retrieval
├── sample_docs/              # Factual documents for testing & ingestion
│   ├── apollo_space_missions.md
│   ├── cloud_computing_overview.md
│   └── photosynthesis_biology.txt
├── src/
│   └── hybrid_rag/           # Core package source code
│       ├── __init__.py       # Package exports
│       ├── chunker.py        # Configurable chunking with page isolation
│       ├── exceptions.py     # Custom error hierarchy
│       ├── loader.py         # DocumentLoader (PDF, TXT, MD)
│       ├── schema.py         # Typed dataclasses (Document, Chunk, RetrievalResult, FusedResult)
│       └── retrieval/        # Retrieval and fusion subpackage
│           ├── __init__.py
│           ├── bm25.py       # BM25Retriever with regex tokenization
│           ├── dense.py      # DenseRetriever with single-pass in-memory cosine sim
│           └── fusion.py     # Reciprocal Rank Fusion (RRF)
└── tests/                    # Unit and integration test suite
    ├── __init__.py
    ├── test_bm25.py          # Lexical ranking, tokenization & edge case tests
    ├── test_chunker.py       # Chunk boundaries, overlap & page isolation tests
    ├── test_dense.py         # Dense similarity & fake model offline tests
    ├── test_environment.py   # Baseline environment verification test
    ├── test_fusion.py        # Exact RRF mathematical & deduplication tests
    └── test_loader.py        # PDF, TXT, MD loader & failure mode tests
```

---

## Quickstart & Environment Setup

### 1. Prerequisites
- **Python 3.11** (detected at `py -3.11` on Windows)
- **Git**

### 2. Create and Activate Virtual Environment
Open PowerShell in the project directory:

```powershell
# Create virtual environment using Python 3.11 launcher
py -3.11 -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

> **Note for PowerShell execution policy**: If you encounter an execution policy restriction when activating, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`

### 3. Install Dependencies
```powershell
# Upgrade pip and install core requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install project in editable development mode with testing tools
pip install -e ".[dev]"
```

### 4. Configure Environment
```powershell
# Copy the template environment configuration
Copy-Item .env.example .env
```
*(Optional)* Open `.env` and insert your Gemini API Key if you want to test generative answer synthesis in Phase 4. If omitted, the system defaults to pure retrieval & evidence inspection.

### 5. Run the Test Suite
Verify that all unit tests across the ingestion layer pass:

```powershell
pytest -v
```

---

## Quick Example: Ingestion & Chunking (Phase 1)

```python
from hybrid_rag import DocumentLoader, TextChunker

# 1. Load document (PDF, Markdown, or TXT)
doc = DocumentLoader.load("sample_docs/apollo_space_missions.md")
print(f"Loaded: {doc.filename} (Type: {doc.file_type}, Pages: {doc.total_pages})")

# 2. Chunk with sliding window
chunker = TextChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_document(doc)

print(f"Total Chunks: {len(chunks)}")
print(f"First Chunk ID: {chunks[0].chunk_id}")
print(f"Offsets: [{chunks[0].char_start} : {chunks[0].char_end}]")
print(f"Page Attribution: {chunks[0].page_number}")
print(f"Snippet:\n{chunks[0].text[:100]}...")
```

Expected Output:
```text
Loaded: apollo_space_missions.md (Type: md, Pages: 1)
Total Chunks: 6
First Chunk ID: apollo_space_missions_937f609c_c0
Offsets: [0 : 500]
Page Attribution: None
Snippet:
# Apollo Lunar Exploration Program

## Overview of the Program
The Apollo program was an American hu...
```


---

## Quick Example: Hybrid Retrieval & Fusion (Phase 2)

Run the end-to-end retrieval smoke test in your virtual environment:

```powershell
python examples/demo_retrieval.py
```

Or query with a custom prompt:
```powershell
python examples/demo_retrieval.py --query "Which enzyme fixes carbon dioxide during the Calvin cycle?"
```

---

## Current Status: Phase 2 Completed
- [x] Phase 0: Workspace inspection, scaffolding, architecture documentation, sample docs.
- [x] Phase 1: Robust document loading (`PDF`, `TXT`, `MD`), typed dataclasses, and metadata-preserving chunking.
- [x] Phase 2: Dual retrieval engines (BM25 lexical search, Dense Bi-Encoder cosine similarity) and Reciprocal Rank Fusion (RRF).
- [ ] *Next: Phase 3 — Cross-Encoder Precision Reranking (`ms-marco-MiniLM-L6-v2`).*

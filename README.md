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
├── app.py                    # Top-level Streamlit entry point
├── evaluation/
│   ├── README.md             # Benchmark description and metrics guide
│   ├── benchmark_dataset.json# Evaluation queries, targets & ground truths
│   ├── benchmark_results.json# Empirical benchmark run results
│   ├── evaluate.py           # Automated IR benchmark evaluation script
│   └── relevance_judgments.json# Explicit human-auditable ground-truth mappings
├── examples/                 # Executable scripts and demos
│   ├── demo_retrieval.py     # Live demonstration of Lexical, Dense, and RRF retrieval
│   ├── demo_reranker.py      # Live demonstration of Cross-Encoder precision reranking
│   └── demo_pipeline.py      # Live demonstration of end-to-end grounded RAG pipeline
├── sample_docs/              # Factual documents for testing & ingestion
│   ├── apollo_space_missions.md
│   ├── cloud_computing_overview.md
│   └── photosynthesis_biology.txt
├── src/
│   └── hybrid_rag/           # Core package source code
│       ├── __init__.py       # Package exports
│       ├── chunker.py        # Configurable chunking with page isolation
│       ├── evaluation.py     # Pure IR metrics (Recall@k, RR, MRR)
│       ├── exceptions.py     # Custom error hierarchy
│       ├── generator.py      # GroundedAnswerGenerator with Gemini API & safe fallback
│       ├── loader.py         # DocumentLoader (PDF, TXT, MD)
│       ├── pipeline.py       # HybridRAGPipeline (orchestrating ingestion to generation)
│       ├── reranker.py       # CrossEncoderReranker (second-stage cross-attention)
│       ├── schema.py         # Typed dataclasses (Document, Chunk, Results, SourcePassage, PipelineResponse)
│       ├── retrieval/        # Retrieval and fusion subpackage
│       │   ├── __init__.py
│       │   ├── bm25.py       # BM25Retriever with regex tokenization
│       │   ├── dense.py      # DenseRetriever with single-pass in-memory cosine sim
│       │   └── fusion.py     # Reciprocal Rank Fusion (RRF)
│       └── ui/               # Streamlit user interface
│           ├── __init__.py
│           └── app.py        # Interactive web application
└── tests/                    # Unit and integration test suite
    ├── __init__.py
    ├── test_bm25.py          # Lexical ranking, tokenization & edge case tests
    ├── test_chunker.py       # Chunk boundaries, overlap & page isolation tests
    ├── test_dense.py         # Dense similarity & fake model offline tests
    ├── test_environment.py   # Baseline environment verification test
    ├── test_evaluation.py    # Metric mathematical & boundary tests
    ├── test_fusion.py        # Exact RRF mathematical & deduplication tests
    ├── test_generator.py     # Grounded synthesis, citation validation & fallback tests
    ├── test_loader.py        # PDF, TXT, MD loader & failure mode tests
    ├── test_pipeline.py      # End-to-end pipeline ingestion, index reuse & provenance tests
    ├── test_reranker.py      # Cross-encoder scoring, ranking & fake model tests
    └── test_ui.py            # Streamlit AppTest headless UI tests
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

## Quick Example: Cross-Encoder Precision Reranking (Phase 3)

Run the two-stage hybrid retrieval and reranking demonstration:

```powershell
python examples/demo_reranker.py
```

Or query with custom candidate pool and top-k parameters:
```powershell
python examples/demo_reranker.py --query "What rocket was used to launch the Apollo missions, and who led its engineering?" --candidate-pool 6 --top-k 3
```

### Important Reranking Concepts
- **Two-Stage Architecture**: Reranking is applied strictly to a pre-filtered candidate pool from Stage 1 (RRF fusion), not the entire corpus. This balances the high $O(K \cdot L^2)$ computational complexity of cross-attention with retrieval latency.
- **Joint Token Interactions**: While Bi-Encoders compress text into fixed vectors independently, Cross-Encoders attend over all query and document tokens jointly via all-to-all attention layers.
- **Uncalibrated Scores**: Cross-Encoder scores are unbounded relevance logits (higher = more relevant). They are *not* calibrated probabilities and should not be interpreted as percentage confidences.
- **Empirical Validation Required**: Adding a reranker shifts candidate ordering according to the model's relevance training, but whether it improves real-world task accuracy must be empirically validated on a benchmark dataset. See the [Phase 5 quantitative evaluation](#quantitative-evaluation-benchmark-phase-5) below for measured results on the included 8-query benchmark.

---

## Quick Example: Integrated Grounded Pipeline (Phase 4)

Run the end-to-end pipeline with the default Evidence Viewer mode (no API key required):

```powershell
python examples/demo_pipeline.py
```

Or query with an active Gemini API key:
```powershell
# Set your API key in your session environment (never hard-code or commit secrets)
$env:GEMINI_API_KEY = "your-actual-gemini-api-key"

python examples/demo_pipeline.py --query "What rocket was used to launch the Apollo missions, and who led its engineering?"
```

### Important Generation & Grounding Concepts
- **Evidence Viewer Fallback**: If no Gemini API key is configured or an API error occurs, answer generation is safely bypassed without crashing. The pipeline outputs structured evidence passages labeled `[S1]`, `[S2]`, etc., making it clear that an API-backed synthesis was not generated.
- **Generation vs. Retrieval**: Retrieval engines score how closely text passages match a query lexically or semantically. High retrieval or reranker scores indicate high passage relevance, but do **not** guarantee that an LLM will interpret the facts correctly or generate an error-free answer.
- **Citation Attribution & Limitations**: The generator prompts the LLM to ground its claims exclusively in the assigned sources and cite them via bracketed tags (e.g., `[S1]`). However, **citations do not guarantee factual correctness**; models may occasionally misattribute evidence or produce subtle inaccuracies even when citing a source. Citations are untrusted until verified against source text.

---

## Interactive Streamlit Web Application (Phase 5)

Launch the web application locally:

```powershell
streamlit run app.py
```

The application provides:
- **Multiple Document Upload**: Supports `.pdf`, `.txt`, and `.md` with automatic character sliding-window chunking.
- **In-Memory Session Isolation**: Uploaded files and index structures are held strictly in the user's active session state and cleaned up after processing.
- **One-Click Sample Loader**: Click **🚀 Load Samples** to index sample Apollo, Biology, and Cloud documents without uploading manual files.
- **Dual-Mode Synthesis**: Operates in **Grounded Gemini Synthesis** mode when `GEMINI_API_KEY` is provided, or gracefully defaults to **Evidence Viewer Mode** when running offline.
- **Provenance Inspection**: View exact source rankings across BM25, Dense similarity, RRF score, and Cross-Encoder relevance logits.

---

## Quantitative Evaluation Benchmark (Phase 5)

Run the automated evaluation benchmark across the test corpus:

```powershell
python evaluation/evaluate.py
```

### Empirical Results (8 Domain Queries, 20 Chunks):

| Retrieval Configuration | Recall@1 | Recall@3 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 (Lexical Only)** | 0.5000 | 0.8750 | 0.8750 | 0.7500 |
| **Dense (Bi-Encoder Only)** | 0.3750 | 0.8750 | **1.0000** | 0.6771 |
| **RRF Hybrid (k=60)** | 0.3750 | 0.8750 | 0.8750 | 0.6667 |
| **Cross-Encoder Reranked** | **0.5000** | 0.8750 | 0.8750 | **0.7500** |

*Detailed benchmark methodology, metric formulas, and query-level relevance judgments are documented in [`evaluation/README.md`](evaluation/README.md).*

---

## Current Status: Phase 5 Completed (Production Ready)
- [x] Phase 0: Workspace inspection, scaffolding, architecture documentation, sample docs.
- [x] Phase 1: Robust document loading (`PDF`, `TXT`, `MD`), typed dataclasses, and metadata-preserving chunking.
- [x] Phase 2: Dual retrieval engines (BM25 lexical search, Dense Bi-Encoder cosine similarity) and Reciprocal Rank Fusion (RRF).
- [x] Phase 3: Cross-Encoder precision reranking (`cross-encoder/ms-marco-MiniLM-L6-v2`) with rank provenance tracking.
- [x] Phase 4: Grounded answer generation (Gemini API with strict grounding), citation validation, and safe Evidence Viewer fallback.
- [x] Phase 5: Interactive Streamlit web application (`app.py`), quantitative evaluation suite (`evaluation/evaluate.py`), and 59 automated tests.




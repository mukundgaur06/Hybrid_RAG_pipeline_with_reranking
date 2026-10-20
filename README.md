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
├── sample_docs/              # Factual documents for testing & ingestion
│   ├── apollo_space_missions.md
│   ├── cloud_computing_overview.md
│   └── photosynthesis_biology.txt
├── src/
│   └── hybrid_rag/           # Core package source code
│       └── __init__.py
└── tests/                    # Unit and integration test suite
    ├── __init__.py
    └── test_environment.py   # Baseline environment verification test
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

### 5. Run Baseline Tests
Verify that your environment and file structure match Phase 0 prerequisites:

```powershell
python tests/test_environment.py
```
Or, once `pytest` is installed in your `.venv`:
```powershell
pytest -v
```

---

## Current Status: Phase 0 Completed
- [x] Workspace inspection and environment audit.
- [x] Architectural documentation and system diagram.
- [x] Project scaffolding and repository hygiene.
- [x] Sample factual test documents.
- [x] Evaluation benchmark dataset.
- [x] Concept glossary and implementation roadmap.
- [ ] *Next: Phase 1 — Document Ingestion and Metadata-Preserving Chunking.*

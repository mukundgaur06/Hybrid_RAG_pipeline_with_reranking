# HybridRAG: Phase-by-Phase Implementation Roadmap

This roadmap organizes the development of HybridRAG into self-contained, verifiable phases. Each phase defines clear entry criteria, deliverables, automated test requirements, and interview discussion topics.

---

## Phase 0: Scaffolding, Architecture & Environment (Current Phase)
- [x] Inspect workspace and check Python/Git environment.
- [x] Define end-to-end architecture and data flow in `docs/ARCHITECTURE.md`.
- [x] Establish beginner-friendly engineering glossary in `docs/GLOSSARY.md`.
- [x] Configure repository hygiene (`.gitignore`, `.env.example`, `requirements.txt`, `pyproject.toml`).
- [x] Create original factual sample documents for testing (`sample_docs/`).
- [x] Create initial evaluation benchmark schema (`evaluation/`).
- [x] Add baseline environment verification test (`tests/test_environment.py`).
- **Checkpoint**: Safe scaffolding complete; no heavy models installed or fake RAG functions committed.

---

## Phase 1: Document Parsing & Metadata-Preserving Chunking (Completed)
- [x] Implement `src/hybrid_rag/loader.py` supporting PDF, TXT, and Markdown.
- [x] Implement `src/hybrid_rag/chunker.py` with sliding window, overlap, and per-page boundary isolation.
- [x] Establish typed schema (`Document`, `DocumentPage`, `Chunk`) in `src/hybrid_rag/schema.py`.
- [x] Define custom exceptions (`UnsupportedFileTypeError`, `EmptyDocumentError`, `DocumentParsingError`) in `src/hybrid_rag/exceptions.py`.
- [x] Full unit test coverage in `tests/test_loader.py` and `tests/test_chunker.py`.
- **Checkpoint**: Robust ingestion layer complete; verified with 17 automated tests.

---

## Phase 2: Dual Retrieval Engines & Reciprocal Rank Fusion (Completed)
- [x] Implement `src/hybrid_rag/retrieval/bm25.py`: Consistent regex tokenization and BM25Okapi scoring.
- [x] Implement `src/hybrid_rag/retrieval/dense.py`: Single-pass embedding encoding, unit normalization, and NumPy cosine similarity.
- [x] Implement `src/hybrid_rag/retrieval/fusion.py`: Scale-invariant Reciprocal Rank Fusion (RRF) with configurable $k=60$ and deterministic tie-breaking.
- [x] Add result dataclasses (`RetrievalResult`, `FusedResult`) in `src/hybrid_rag/schema.py`.
- [x] Create 100% offline unit tests (`tests/test_bm25.py`, `tests/test_dense.py`, `tests/test_fusion.py`) with a `FakeEmbeddingModel` test double.
- [x] Add live demonstration and smoke test in `examples/demo_retrieval.py`.
- **Checkpoint**: Retrieval and fusion engines complete; verified with 32 automated tests.

---

## Phase 3: Hybrid Fusion (RRF) & Cross-Encoder Reranking
- **Objective**: Fuse disparate retrieval rankings and apply deep transformer cross-attention to re-order the top candidates.
- **Key Deliverables**:
  - `src/hybrid_rag/retrieval/fusion.py`: Scale-invariant Reciprocal Rank Fusion (RRF) combining BM25 and Dense result ranks.
  - `src/hybrid_rag/reranker.py`: Cross-Encoder reranking using `cross-encoder/ms-marco-MiniLM-L6-v2` taking query-document pairs.
- **Testing**:
  - Fusion unit test verifying RRF score calculation with standard $k=60$.
  - End-to-end reranking test comparing pre-rerank and post-rerank candidate order on ambiguous queries.
- **Interview Focus**: Bi-Encoder vs. Cross-Encoder compute trade-offs; why score normalization fails across heterogeneous retrievers and why RRF solves it.

---

## Phase 4: Grounded Synthesis & Citation Engine
- **Objective**: Synthesize concise answers backed by verifiable citations using Gemini API, with a graceful fallback if no API key is provided.
- **Key Deliverables**:
  - `src/hybrid_rag/generator.py`: Prompt engineering for strict factual grounding; API wrapper using `google-genai` SDK.
  - `src/hybrid_rag/pipeline.py`: Unified pipeline coordinating document loading, indexing, hybrid retrieval, reranking, and generation.
  - Fallback logic: "Evidence Viewer" mode presenting ranked evidence chunks with confidence scores and snippet highlights when no API key exists.
- **Testing**:
  - Mocked LLM tests checking prompt construction and citation extraction.
  - Fallback test ensuring retrieval and evidence output work completely offline without network calls or API keys.
- **Interview Focus**: Hallucination mitigation techniques, prompt injection defenses, citation attribution, and offline resiliency.

---

## Phase 5: Streamlit Application, Retrieval Evaluation & Benchmarking
- **Objective**: Build an interactive web UI and run an automated evaluation suite calculating Recall@k and MRR.
- **Key Deliverables**:
  - `src/hybrid_rag/ui/app.py`: Streamlit application featuring document upload, query input, answer generation, expandable citations, and score inspection tabs (BM25 vs. Dense vs. RRF vs. Reranker).
  - `evaluation/evaluate.py`: Automated evaluation script measuring Recall@1, Recall@5, and MRR against `evaluation/benchmark_dataset.json`.
- **Testing**:
  - Integration test evaluating pipeline performance on the benchmark dataset.
- **Interview Focus**: Quantitative RAG evaluation methodology; interpreting retrieval metrics; production monitoring and latency profiling.

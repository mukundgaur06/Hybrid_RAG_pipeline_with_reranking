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
  - [x] `src/hybrid_rag/retrieval/fusion.py`: Scale-invariant Reciprocal Rank Fusion (RRF) combining BM25 and Dense result ranks.
  - [x] `src/hybrid_rag/reranker.py`: Cross-Encoder reranking using `cross-encoder/ms-marco-MiniLM-L6-v2` taking query-document pairs.
  - [x] `src/hybrid_rag/schema.py`: Added `RerankedResult` dataclass preserving complete pre-rerank provenance.
  - [x] `src/hybrid_rag/__init__.py`: Exported `CrossEncoderReranker` and `RerankedResult`.
  - [x] `tests/test_reranker.py`: 100% offline unit tests verifying sorting, bounds, tie-breaking, metadata preservation, and error handling with `FakeCrossEncoder`.
  - [x] `examples/demo_reranker.py`: Interactive demonstration comparing Stage 1 RRF candidates with Stage 2 Cross-Encoder results.
- **Testing**:
  - [x] 10 dedicated offline reranker unit tests passing.
  - [x] Full automated test suite (42 tests) passing without regressions.
- **Checkpoint**: Cross-Encoder precision reranker complete; verified with 42 automated tests and smoke demonstration script.
- **Interview Focus**: Bi-Encoder vs. Cross-Encoder compute trade-offs; why score normalization fails across heterogeneous retrievers and why RRF solves it; uncalibrated logits vs. probabilities; multi-stage retrieval architecture.


---

## Phase 4: Grounded Synthesis & Citation Engine
- **Objective**: Synthesize concise answers backed by verifiable citations using Gemini API, with a graceful fallback if no API key is provided.
- **Key Deliverables**:
  - [x] `src/hybrid_rag/generator.py`: Prompt engineering for strict factual grounding; API wrapper using `google-genai` SDK with injected client seam.
  - [x] `src/hybrid_rag/pipeline.py`: Unified pipeline coordinating document loading, indexing, hybrid retrieval, reranking, and generation.
  - [x] `src/hybrid_rag/schema.py`: Added `SourcePassage` and `PipelineResponse` data structures for citation mapping and safe provenance tracking.
  - [x] `src/hybrid_rag/__init__.py`: Exported `GroundedAnswerGenerator`, `HybridRAGPipeline`, `SourcePassage`, and `PipelineResponse`.
  - [x] Fallback logic: "Evidence Viewer" mode presenting ranked evidence chunks with source identifiers and metadata when no API key exists or on API error.
  - [x] Citation validation: Regex citation parsing cross-checked against assigned source IDs to separate valid citations from hallucinated citations.
  - [x] `examples/demo_pipeline.py`: Interactive end-to-end command-line demonstration supporting both API-backed generation and Evidence Viewer fallback.
- **Testing**:
  - [x] 7 dedicated generator unit tests (`tests/test_generator.py`) checking prompt formatting, citation validation, missing-key fallback, and API error safety.
  - [x] 4 dedicated pipeline integration tests (`tests/test_pipeline.py`) verifying index reuse, provenance preservation, and error handling.
  - [x] Full automated test suite (53 tests) passing 100% offline without network calls or API keys.
- **Checkpoint**: Full end-to-end pipeline complete; verified with 53 automated tests and CLI demonstration script.
- **Interview Focus**: Hallucination mitigation techniques; difference between retrieval scores and answer correctness; citation attribution limitations; offline resiliency and graceful degradation.

---

## Phase 5: Streamlit Application, Retrieval Evaluation & Benchmarking
- **Objective**: Build an interactive web UI and run an automated evaluation suite calculating Recall@k and MRR.
- **Key Deliverables**:
  - [x] `src/hybrid_rag/ui/app.py` & `app.py`: Sleek dark Streamlit application with multi-file upload (.pdf, .txt, .md), session isolation, sample loader, Evidence Viewer mode, and multi-stage provenance inspection tabs.
  - [x] `src/hybrid_rag/evaluation.py`: Exact mathematical definitions and implementations for Recall@k, Reciprocal Rank, and Mean Reciprocal Rank (MRR).
  - [x] `evaluation/relevance_judgments.json`: Explicit, human-auditable ground-truth chunk relevance mappings for all benchmark queries.
  - [x] `evaluation/evaluate.py`: Automated reproducible evaluation suite measuring Recall@1, Recall@3, Recall@5, and MRR across BM25, Dense, RRF, and Cross-Encoder reranking.
  - [x] `evaluation/benchmark_results.json`: Detailed empirical benchmark run outputs.
- **Testing**:
  - [x] 4 offline unit tests for evaluation metrics (`tests/test_evaluation.py`).
  - [x] 2 offline UI tests using `streamlit.testing.v1.AppTest` (`tests/test_ui.py`).
  - [x] Full automated test suite (59 tests) passing 100% offline without network calls or API keys.
- **Checkpoint**: Production-ready HybridRAG system complete; validated across all 5 phases with 59 automated tests, quantitative benchmark evaluations, and an interactive Streamlit UI.
- **Interview Focus**: Quantitative RAG evaluation methodology; interpreting retrieval metrics; production monitoring and latency profiling; session isolation vs. shared model resource caching.


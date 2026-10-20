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

## Phase 1: Document Parsing & Metadata-Preserving Chunking
- **Objective**: Implement clean document ingestion and chunking without losing source provenance.
- **Key Deliverables**:
  - `src/hybrid_rag/loader.py`: Modular readers for PDF (`PyMuPDF`), Markdown, and plaintext.
  - `src/hybrid_rag/chunker.py`: Overlapping chunking engine producing typed `Chunk` dataclasses with `chunk_id`, `document_name`, `page_number`, and `text`.
- **Testing**:
  - Unit tests verifying chunk length constraints and overlap consistency.
  - Test verifying PDF page number tracking matches actual page counts.
- **Interview Focus**: Why fixed-size chunking with overlap prevents context truncation at token boundaries; handling multi-modal or structured document pitfalls.

---

## Phase 2: Dual Retrieval Engines (BM25 & Dense Bi-Encoder)
- **Objective**: Implement lexical and dense vector search independently in pure Python and NumPy.
- **Key Deliverables**:
  - `src/hybrid_rag/retrieval/bm25.py`: Corpus tokenization, term-frequency indexing, and BM25Okapi scoring using `rank-bm25`.
  - `src/hybrid_rag/retrieval/dense.py`: Embedding computation with `sentence-transformers` (`all-MiniLM-L6-v2`) and vectorized cosine similarity using NumPy matrix multiplication.
- **Testing**:
  - Keyword search test where BM25 finds exact technical terms that dense search misses.
  - Semantic paraphrase test where dense search matches synonymous concepts that share no common keywords with the document.
- **Interview Focus**: Inverted indices vs. vector index math; time and space complexity of cosine similarity across $N$ document vectors.

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

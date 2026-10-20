# HybridRAG: Evaluation & Benchmark Framework

## Purpose
A rigorous RAG system cannot be evaluated solely by "eyeballing" generated answers. Machine learning engineering requires quantitative, reproducible metrics to validate that retrieval changes (such as adjusting chunk size, modifying the embedding model, or adding a reranker) genuinely improve document relevance.

## Evaluation Methodology
This benchmark evaluates retrieval performance across multiple search configurations:
1. **Lexical Only (BM25)**
2. **Dense Only (Bi-Encoder embeddings + Cosine similarity)**
3. **Hybrid (BM25 + Dense merged via Reciprocal Rank Fusion)**
4. **Hybrid + Cross-Encoder Reranking**

### Metrics Defined
- **Recall@k**: Proportion of queries where at least one ground-truth relevant chunk was present in the top $k$ retrieved results ($k \in \{1, 3, 5\}$).
- **MRR (Mean Reciprocal Rank)**: The average of the reciprocal ranks of the first relevant chunk retrieved across all test queries. Measures how close to rank 1 the system places the correct evidence.

## Dataset Structure (`benchmark_dataset.json`)
Each evaluation entry contains:
- `query_id`: Unique identifier (e.g., `q01`).
- `query`: The natural language question.
- `source_document`: The sample document containing the ground truth.
- `relevant_keywords`: Essential key terms that must appear in the retrieved evidence chunk.
- `ground_truth_answer`: Factual reference answer for validation.
- `query_type`: Classification of query difficulty (`keyword_specific`, `semantic_paraphrase`, `multi_hop`, `numerical_fact`).

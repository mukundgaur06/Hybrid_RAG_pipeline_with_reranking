# HybridRAG: Quantitative Evaluation & Benchmark Framework

## 1. Purpose
A rigorous RAG system cannot be evaluated solely by "eyeballing" generated answers. Machine learning engineering requires quantitative, reproducible metrics to validate how retrieval stages (lexical BM25, dense embeddings, reciprocal rank fusion, and cross-encoder reranking) perform against ground-truth evidence.

---

## 2. Methodology & Mathematical Metrics

This benchmark evaluates retrieval performance across four progressive configurations:
1. **BM25 Lexical Retrieval**: Term frequency and inverse document frequency with length normalization.
2. **Dense Semantic Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` dense vectors with cosine similarity.
3. **RRF Hybrid Fusion**: Reciprocal Rank Fusion ($k=60$) blending ordinal ranks from BM25 and Dense branches.
4. **Cross-Encoder Precision Reranking**: `cross-encoder/ms-marco-MiniLM-L6-v2` scoring joint `(query, passage)` pairs over the fused candidate pool.

### Metrics:
- **Recall@k**:
  $$\text{Recall@}k = \frac{|\text{Retrieved[:}k\text{]} \cap \text{Relevant}|}{|\text{Relevant}|}$$
  Measures whether the necessary ground-truth evidence chunk appears within the top $k$ positions ($k \in \{1, 3, 5\}$).
- **Mean Reciprocal Rank (MRR)**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  where $\text{rank}_i$ is the 1-based rank position of the first relevant chunk retrieved for query $i$ (or $0$ if unretrieved).

---

## 3. Dataset & Explicit Ground-Truth Judgments

The benchmark evaluates 8 curated queries across three domain corpora in `sample_docs/`:
- **Apollo Lunar Missions** (`apollo_space_missions.md`): Space engineering, Apollo 11, Apollo 13 abort.
- **Photosynthesis & Cellular Biology** (`photosynthesis_biology.txt`): Light-dependent photolysis, Calvin cycle RuBisCO carboxylation.
- **Cloud Computing & Architecture** (`cloud_computing_overview.md`): SLA uptime mathematics, Cache-Aside pattern.

Ground-truth chunk judgments are explicitly declared in `evaluation/relevance_judgments.json`. Each entry maps a query ID to verified chunk IDs with documented rationale. Relevance is never generated from the system's own predictions.

---

## 4. How to Run the Benchmark

Execute the benchmark using the project's virtual environment:

```powershell
python evaluation/evaluate.py
```

This runs all 8 queries across the 4 configurations and exports detailed per-query rankings to `evaluation/benchmark_results.json`.

---

## 5. Empirical Benchmark Results

Measured across all 8 queries on the indexed corpus (20 chunks, chunk size 500, overlap 100):

| Retrieval Configuration | Recall@1 | Recall@3 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 (Lexical Only)** | 0.5000 | 0.8750 | 0.8750 | 0.7500 |
| **Dense (Bi-Encoder Only)** | 0.3750 | 0.8750 | **1.0000** | 0.6771 |
| **RRF Hybrid (k=60)** | 0.3750 | 0.8750 | 0.8750 | 0.6667 |
| **Cross-Encoder Reranked** | **0.5000** | 0.8750 | 0.8750 | **0.7500** |

---

## 6. Engineering Interpretation & Trade-offs

> **Important caveat**: All findings below are measured on a small benchmark of 8 queries and 20 chunks across 3 documents. Results on this scale are illustrative and should not be extrapolated to general-domain retrieval performance. Broader validation on larger corpora (e.g., MS MARCO, BEIR) is required before drawing general conclusions.

1. **Dense Retrieval Achieves the Highest Recall@5 (1.0000)**:
   - Dense embeddings captured semantic paraphrasing on complex biology and cloud questions where exact terms differed, ensuring every query had its correct evidence chunk within the top 5 candidates. No other configuration matched this on this benchmark.
2. **Cross-Encoder Reranking Improves Precision Relative to RRF (MRR: 0.6667 → 0.7500, Recall@1: 0.3750 → 0.5000)**:
   - When reranking the RRF candidate pool, the Cross-Encoder's all-to-all attention promoted relevant chunks that were previously buried at Rank 2 or 3 into Rank 1 (e.g. promoting the Saturn V / Wernher von Braun evidence chunk). This improvement is relative to RRF fusion on this benchmark.
3. **Cross-Encoder Ties BM25 on This Benchmark**:
   - Cross-Encoder reranking (MRR 0.7500, Recall@1 0.5000) matches BM25 lexical retrieval exactly on this 8-query benchmark. This reflects that BM25's exact token matching remains very strong for the technical terminology present in these queries. Hybrid fusion (RRF) did not surpass individual methods on this small corpus.
4. **BM25 Remains Critical for Technical Terms**:
   - Exact numerical queries (e.g., "34.5 meganewtons", "52.6 minutes") and chemical equations benefit heavily from BM25's exact token matching. This contributes to its strong performance on this technically-worded benchmark.
5. **Hybrid Retrieval Does Not Universally Outperform Individual Methods**:
   - On this benchmark, RRF fusion (MRR 0.6667) scores lower than either BM25 or Cross-Encoder. Hybrid fusion's advantage typically emerges on larger, more diverse corpora where BM25 and dense retrieval have complementary failure modes. These results do not support a universal claim that hybrid retrieval is superior.
6. **Benchmark Limitations**:
   - The test corpus contains only 20 chunks across 3 documents and 8 curated queries. Results at this scale may not represent performance on larger or more diverse collections. **Broader validation on standard IR benchmarks is needed before drawing general conclusions** about the relative merit of these retrieval configurations.

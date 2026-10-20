"""Quantitative Information Retrieval Evaluation Suite for HybridRAG.

Evaluates and compares:
1. Lexical retrieval (BM25)
2. Dense semantic retrieval (Bi-Encoder embeddings + cosine similarity)
3. Hybrid rank fusion (Reciprocal Rank Fusion - RRF k=60)
4. Two-stage precision reranking (Cross-Encoder ms-marco-MiniLM-L6-v2)

Metrics:
- Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from hybrid_rag import (
    BM25Retriever,
    Chunk,
    CrossEncoderReranker,
    DenseRetriever,
    DocumentLoader,
    TextChunker,
    reciprocal_rank_fusion,
)
from hybrid_rag.evaluation import (
    calculate_mrr,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)


def load_sample_chunks(sample_dir: Path) -> List[Chunk]:
    """Ingest and chunk all sample documents."""
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    all_chunks: List[Chunk] = []

    files = sorted([f for f in sample_dir.glob("*.*") if f.suffix.lower() in {".md", ".txt", ".pdf"}])
    for f in files:
        doc = loader.load(f)
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    return all_chunks


def run_benchmark(
    dataset_path: Path,
    judgments_path: Path,
    sample_dir: Path,
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, Any]:
    """Execute evaluation across all benchmark queries."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark_queries = json.load(f)

    with open(judgments_path, "r", encoding="utf-8") as f:
        judgments_data = json.load(f)["judgments"]

    print("=" * 80)
    print("HYBRID RAG: RETRIEVAL & RERANKING EVALUATION BENCHMARK")
    print("=" * 80)
    print(f"Total benchmark queries: {len(benchmark_queries)}")

    # 1. Ingest and build indexes once
    print("\n[1/3] Ingesting sample documents and indexing chunks in memory...")
    chunks = load_sample_chunks(sample_dir)
    print(f"      Total indexed chunks: {len(chunks)}")

    print("[2/3] Initializing retrieval engines (BM25, Dense, Cross-Encoder)...")
    bm25 = BM25Retriever(chunks=chunks)
    dense = DenseRetriever(chunks=chunks)
    reranker = CrossEncoderReranker()

    methods = ["BM25", "Dense", "RRF_Hybrid", "Cross_Encoder"]
    max_k = max(k_values)

    # Store retrieved chunk IDs per method for all queries
    retrieved_by_method: Dict[str, List[List[str]]] = {m: [] for m in methods}
    relevant_by_query: List[Set[str]] = []
    per_query_records: List[Dict[str, Any]] = []

    print(f"\n[3/3] Executing retrieval and ranking over {len(benchmark_queries)} queries...")

    for item in benchmark_queries:
        qid = item["query_id"]
        query_text = item["query"]
        relevant_chunk_ids = set(judgments_data[qid]["relevant_chunk_ids"])
        relevant_by_query.append(relevant_chunk_ids)

        # 1. BM25
        bm25_res = bm25.retrieve(query_text, top_k=max_k)
        bm25_ids = [r.chunk.chunk_id for r in bm25_res]
        retrieved_by_method["BM25"].append(bm25_ids)

        # 2. Dense
        dense_res = dense.retrieve(query_text, top_k=max_k)
        dense_ids = [r.chunk.chunk_id for r in dense_res]
        retrieved_by_method["Dense"].append(dense_ids)

        # 3. RRF Hybrid
        fused_res = reciprocal_rank_fusion(bm25_res, dense_res, k=60, top_k=max_k)
        fused_ids = [r.chunk.chunk_id for r in fused_res]
        retrieved_by_method["RRF_Hybrid"].append(fused_ids)

        # 4. Cross-Encoder Reranking
        reranked_res = reranker.rerank(query_text, candidates=fused_res, top_k=max_k)
        rerank_ids = [r.chunk.chunk_id for r in reranked_res]
        retrieved_by_method["Cross_Encoder"].append(rerank_ids)

        # Record individual query metrics for auditing
        record = {
            "query_id": qid,
            "query": query_text,
            "relevant_chunk_ids": list(relevant_chunk_ids),
            "retrieved": {
                "BM25": bm25_ids,
                "Dense": dense_ids,
                "RRF_Hybrid": fused_ids,
                "Cross_Encoder": rerank_ids,
            },
        }
        per_query_records.append(record)

    # Calculate aggregate metrics
    results: Dict[str, Dict[str, float]] = {}
    for m in methods:
        m_results: Dict[str, float] = {}
        for k in k_values:
            recalls = [
                calculate_recall_at_k(ret, rel, k=k)
                for ret, rel in zip(retrieved_by_method[m], relevant_by_query)
            ]
            m_results[f"Recall@{k}"] = float(sum(recalls)) / float(len(recalls))

        m_results["MRR"] = calculate_mrr(retrieved_by_method[m], relevant_by_query)
        results[m] = m_results

    # Print Formatted Table
    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK RESULTS")
    print("=" * 80)
    header = f"{'Retrieval Configuration':<25} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<10}"
    print(header)
    print("-" * 80)
    for m in methods:
        row = (
            f"{m:<25} | "
            f"{results[m]['Recall@1']:<10.4f} | "
            f"{results[m]['Recall@3']:<10.4f} | "
            f"{results[m]['Recall@5']:<10.4f} | "
            f"{results[m]['MRR']:<10.4f}"
        )
        print(row)
    print("=" * 80)

    return {
        "summary": results,
        "per_query": per_query_records,
        "k_values": k_values,
        "total_queries": len(benchmark_queries),
    }


def main():
    parser = argparse.ArgumentParser(description="HybridRAG IR Evaluation Benchmark")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/benchmark_results.json",
        help="Path to save benchmark JSON output",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "evaluation" / "benchmark_dataset.json"
    judgments_path = base_dir / "evaluation" / "relevance_judgments.json"
    sample_dir = base_dir / "sample_docs"

    results = run_benchmark(dataset_path, judgments_path, sample_dir)

    out_path = base_dir / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed evaluation results exported to: {out_path}\n")


if __name__ == "__main__":
    main()

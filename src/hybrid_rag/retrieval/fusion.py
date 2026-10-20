"""Reciprocal Rank Fusion (RRF) for combining lexical and semantic retrieval candidates.

Merges candidate rankings without relying on score normalization, using the scale-invariant
formula: RRF_Score(d) = SUM ( 1 / (k + rank_m(d)) ).
"""

from typing import Dict, List, Optional

from hybrid_rag.schema import Chunk, FusedResult, RetrievalResult


def reciprocal_rank_fusion(
    bm25_results: List[RetrievalResult],
    dense_results: List[RetrievalResult],
    k: int = 60,
    top_k: int = 10,
) -> List[FusedResult]:
    """Fuse ranked candidate lists from BM25 and Dense retrieval using RRF.

    Reciprocal Rank Fusion operates strictly on ordinal ranks rather than raw scores.
    This guarantees scale-invariance when combining unbounded lexical scores (BM25)
    with bounded geometric scores (cosine similarity).

    Args:
        bm25_results: Ranked candidates from BM25 lexical retrieval.
        dense_results: Ranked candidates from Dense vector retrieval.
        k: Smoothing constant balancing the influence of high vs. low ranks.
           Standard default in information retrieval literature is 60.
        top_k: Maximum number of fused results to return.

    Returns:
        List of FusedResult objects sorted by RRF score descending.

    Raises:
        ValueError: If k is not a positive integer.
    """
    if k <= 0:
        raise ValueError(f"RRF smoothing constant k must be positive, got {k}")
    if top_k <= 0:
        return []

    # Map chunk_id to intermediate aggregation record
    # Structure: {
    #   "chunk": Chunk,
    #   "rrf_score": float,
    #   "bm25_rank": Optional[int],
    #   "bm25_score": Optional[float],
    #   "dense_rank": Optional[int],
    #   "dense_score": Optional[float]
    # }
    candidates: Dict[str, Dict] = {}

    # 1. Accumulate BM25 contributions
    for item in bm25_results:
        cid = item.chunk.chunk_id
        contribution = 1.0 / (k + item.rank)

        if cid not in candidates:
            candidates[cid] = {
                "chunk": item.chunk,
                "rrf_score": contribution,
                "bm25_rank": item.rank,
                "bm25_score": item.score,
                "dense_rank": None,
                "dense_score": None,
            }
        else:
            candidates[cid]["rrf_score"] += contribution
            candidates[cid]["bm25_rank"] = item.rank
            candidates[cid]["bm25_score"] = item.score

    # 2. Accumulate Dense contributions
    for item in dense_results:
        cid = item.chunk.chunk_id
        contribution = 1.0 / (k + item.rank)

        if cid not in candidates:
            candidates[cid] = {
                "chunk": item.chunk,
                "rrf_score": contribution,
                "bm25_rank": None,
                "bm25_score": None,
                "dense_rank": item.rank,
                "dense_score": item.score,
            }
        else:
            candidates[cid]["rrf_score"] += contribution
            candidates[cid]["dense_rank"] = item.rank
            candidates[cid]["dense_score"] = item.score

    if not candidates:
        return []

    # 3. Sort deterministically: highest RRF score first, tie-break by chunk_id ascending
    sorted_items = sorted(
        candidates.values(),
        key=lambda entry: (-entry["rrf_score"], entry["chunk"].chunk_id),
    )

    # 4. Limit to top_k and build FusedResult objects with 1-based ranks
    fused_results: List[FusedResult] = []
    for rank, entry in enumerate(sorted_items[:top_k], start=1):
        fused_results.append(
            FusedResult(
                chunk=entry["chunk"],
                rrf_score=entry["rrf_score"],
                rank=rank,
                bm25_rank=entry["bm25_rank"],
                bm25_score=entry["bm25_score"],
                dense_rank=entry["dense_rank"],
                dense_score=entry["dense_score"],
            )
        )

    return fused_results

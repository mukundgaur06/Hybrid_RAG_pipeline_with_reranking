"""Unit tests for Reciprocal Rank Fusion (RRF).

Verifies the mathematical correctness of 1 / (k + rank), deduplication,
preservation of source ranks and raw scores, deterministic tie-breaking,
and error handling.
"""

import pytest

from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.schema import Chunk, RetrievalResult


def _create_dummy_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        document_name="doc.txt",
        page_number=1,
        chunk_index=0,
        text=f"Text for {chunk_id}",
        char_start=0,
        char_end=20,
    )


def test_rrf_exact_mathematical_calculation():
    """Verify that RRF computes SUM(1 / (k + rank)) exactly according to theory."""
    c1 = _create_dummy_chunk("chunk_1")
    c2 = _create_dummy_chunk("chunk_2")

    # BM25: chunk_1 is rank 1, chunk_2 is rank 2
    bm25_results = [
        RetrievalResult(chunk=c1, score=12.5, rank=1),
        RetrievalResult(chunk=c2, score=7.2, rank=2),
    ]

    # Dense: chunk_1 is rank 2, chunk_2 is absent
    dense_results = [
        RetrievalResult(chunk=c1, score=0.88, rank=2),
    ]

    k = 60
    fused = reciprocal_rank_fusion(bm25_results, dense_results, k=k, top_k=5)

    assert len(fused) == 2

    # chunk_1 appeared at BM25 rank 1 and Dense rank 2:
    # expected = 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    expected_c1_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert fused[0].chunk.chunk_id == "chunk_1"
    assert fused[0].rank == 1
    assert pytest.approx(fused[0].rrf_score, abs=1e-6) == expected_c1_score
    assert fused[0].bm25_rank == 1
    assert fused[0].bm25_score == 12.5
    assert fused[0].dense_rank == 2
    assert fused[0].dense_score == 0.88

    # chunk_2 appeared only in BM25 at rank 2:
    # expected = 1/(60+2) = 1/62
    expected_c2_score = 1.0 / 62.0
    assert fused[1].chunk.chunk_id == "chunk_2"
    assert fused[1].rank == 2
    assert pytest.approx(fused[1].rrf_score, abs=1e-6) == expected_c2_score
    assert fused[1].bm25_rank == 2
    assert fused[1].bm25_score == 7.2
    assert fused[1].dense_rank is None
    assert fused[1].dense_score is None


def test_rrf_does_not_sum_raw_scores():
    """Verify that raw BM25 scores and cosine scores are never added directly."""
    c_high_bm25 = _create_dummy_chunk("c_high_bm25")
    c_low_bm25 = _create_dummy_chunk("c_low_bm25")

    # c_high_bm25 has a massive raw BM25 score (100.0) at rank 2
    # c_low_bm25 has a lower raw BM25 score (10.0) at rank 1
    bm25_results = [
        RetrievalResult(chunk=c_low_bm25, score=10.0, rank=1),
        RetrievalResult(chunk=c_high_bm25, score=100.0, rank=2),
    ]

    fused = reciprocal_rank_fusion(bm25_results, [], k=60)

    # In rank fusion, Rank 1 MUST beat Rank 2 regardless of raw score magnitudes
    assert fused[0].chunk.chunk_id == "c_low_bm25"
    assert fused[1].chunk.chunk_id == "c_high_bm25"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_rrf_deterministic_tie_breaking():
    """Verify that ties in RRF scores are broken deterministically by chunk_id."""
    c_b = _create_dummy_chunk("chunk_b")
    c_a = _create_dummy_chunk("chunk_a")

    # Each chunk appears once at rank 1 in one retriever
    bm25_results = [RetrievalResult(chunk=c_b, score=5.0, rank=1)]
    dense_results = [RetrievalResult(chunk=c_a, score=0.9, rank=1)]

    fused = reciprocal_rank_fusion(bm25_results, dense_results, k=60)

    assert len(fused) == 2
    assert fused[0].rrf_score == fused[1].rrf_score
    # Should sort alphabetically by chunk_id
    assert fused[0].chunk.chunk_id == "chunk_a"
    assert fused[1].chunk.chunk_id == "chunk_b"


def test_rrf_empty_inputs_and_validation():
    """Verify handling of empty lists and validation on parameter k."""
    # Empty inputs
    assert reciprocal_rank_fusion([], []) == []
    assert reciprocal_rank_fusion([RetrievalResult(_create_dummy_chunk("c1"), 1.0, 1)], [], top_k=0) == []

    # Invalid k
    with pytest.raises(ValueError, match="constant k must be positive"):
        reciprocal_rank_fusion([], [], k=0)

    with pytest.raises(ValueError, match="constant k must be positive"):
        reciprocal_rank_fusion([], [], k=-10)

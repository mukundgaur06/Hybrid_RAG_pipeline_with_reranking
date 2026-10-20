"""Unit tests for CrossEncoderReranker.

Uses an injected test double (FakeCrossEncoder) to provide 100% offline,
deterministic testing of cross-encoder scoring, top-k ranking, metadata
preservation, tie-breaking, and error handling without model downloads.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pytest

from hybrid_rag.reranker import CrossEncoderReranker
from hybrid_rag.schema import Chunk, FusedResult, RetrievalResult


class FakeCrossEncoder:
    """Test double returning controlled scores for (query, passage) pairs."""

    def __init__(
        self,
        score_map: Optional[Dict[Tuple[str, str], float]] = None,
        default_score: float = 0.0,
    ):
        self.score_map = score_map or {}
        self.default_score = default_score
        self.recorded_pairs: List[Tuple[str, str]] = []
        self.recorded_batch_sizes: List[int] = []

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        self.recorded_pairs.extend(pairs)
        self.recorded_batch_sizes.append(batch_size)
        scores = [self.score_map.get(pair, self.default_score) for pair in pairs]
        return np.array(scores, dtype=np.float32)


def _create_sample_fused_candidates() -> List[FusedResult]:
    """Helper to create realistic FusedResult candidates with provenance."""
    c1 = Chunk(
        chunk_id="chunk_001",
        document_id="doc1",
        document_name="apollo.md",
        page_number=1,
        chunk_index=0,
        text="Saturn V rocket was developed under Wernher von Braun.",
        char_start=0,
        char_end=53,
    )
    c2 = Chunk(
        chunk_id="chunk_002",
        document_id="doc1",
        document_name="apollo.md",
        page_number=2,
        chunk_index=1,
        text="The Command Module Columbia splashed down in the Pacific Ocean.",
        char_start=54,
        char_end=117,
    )
    c3 = Chunk(
        chunk_id="chunk_003",
        document_id="doc2",
        document_name="space.txt",
        page_number=None,
        chunk_index=0,
        text="Lunar roving vehicles were used in later Apollo missions.",
        char_start=0,
        char_end=57,
    )

    f1 = FusedResult(
        chunk=c1,
        rrf_score=0.032,
        rank=1,
        bm25_rank=1,
        bm25_score=5.2,
        dense_rank=2,
        dense_score=0.81,
    )
    f2 = FusedResult(
        chunk=c2,
        rrf_score=0.028,
        rank=2,
        bm25_rank=2,
        bm25_score=3.8,
        dense_rank=3,
        dense_score=0.74,
    )
    f3 = FusedResult(
        chunk=c3,
        rrf_score=0.025,
        rank=3,
        bm25_rank=None,
        bm25_score=None,
        dense_rank=1,
        dense_score=0.88,
    )

    return [f1, f2, f3]


def test_reranker_descending_score_ordering_and_ranks():
    """Verify results are sorted by reranker_score descending with 1-based ranks."""
    query = "Who developed the Saturn V rocket?"
    candidates = _create_sample_fused_candidates()

    # Model scores c1 highest, c3 second, c2 lowest
    score_map = {
        (query, candidates[0].chunk.text): 4.5,
        (query, candidates[1].chunk.text): -1.2,
        (query, candidates[2].chunk.text): 2.1,
    }
    fake_model = FakeCrossEncoder(score_map=score_map)
    reranker = CrossEncoderReranker(model=fake_model)

    results = reranker.rerank(query=query, candidates=candidates)

    assert len(results) == 3

    # Rank 1: chunk_001 (score 4.5)
    assert results[0].chunk.chunk_id == "chunk_001"
    assert results[0].rerank_rank == 1
    assert pytest.approx(results[0].reranker_score, abs=1e-4) == 4.5

    # Rank 2: chunk_003 (score 2.1)
    assert results[1].chunk.chunk_id == "chunk_003"
    assert results[1].rerank_rank == 2
    assert pytest.approx(results[1].reranker_score, abs=1e-4) == 2.1

    # Rank 3: chunk_002 (score -1.2)
    assert results[2].chunk.chunk_id == "chunk_002"
    assert results[2].rerank_rank == 3
    assert pytest.approx(results[2].reranker_score, abs=1e-4) == -1.2


def test_reranker_preserves_rrf_and_retrieval_provenance():
    """Verify that RRF ranks, RRF scores, BM25 and Dense metrics are preserved."""
    query = "Apollo mission details"
    candidates = _create_sample_fused_candidates()

    fake_model = FakeCrossEncoder(default_score=1.0)
    reranker = CrossEncoderReranker(model=fake_model)

    results = reranker.rerank(query=query, candidates=candidates)

    res1 = next(r for r in results if r.chunk.chunk_id == "chunk_001")
    assert res1.rrf_score == 0.032
    assert res1.rrf_rank == 1
    assert res1.bm25_rank == 1
    assert res1.bm25_score == 5.2
    assert res1.dense_rank == 2
    assert res1.dense_score == 0.81

    res3 = next(r for r in results if r.chunk.chunk_id == "chunk_003")
    assert res3.rrf_score == 0.025
    assert res3.rrf_rank == 3
    assert res3.bm25_rank is None
    assert res3.bm25_score is None
    assert res3.dense_rank == 1
    assert res3.dense_score == 0.88


def test_reranker_preserves_chunk_metadata():
    """Verify chunk metadata (id, filename, page, text, offsets) is intact."""
    query = "Apollo mission"
    candidates = _create_sample_fused_candidates()

    fake_model = FakeCrossEncoder(default_score=1.0)
    reranker = CrossEncoderReranker(model=fake_model)

    results = reranker.rerank(query=query, candidates=candidates)

    for res, cand in zip(results, candidates):
        assert res.chunk.document_id == cand.chunk.document_id
        assert res.chunk.document_name == cand.chunk.document_name
        assert res.chunk.page_number == cand.chunk.page_number
        assert res.chunk.chunk_index == cand.chunk.chunk_index
        assert res.chunk.text == cand.chunk.text
        assert res.chunk.char_start == cand.chunk.char_start
        assert res.chunk.char_end == cand.chunk.char_end


def test_reranker_top_k_behavior():
    """Verify top_k limiting, top_k exceeding, top_k=None, and non-positive bounds."""
    query = "Test query"
    candidates = _create_sample_fused_candidates()

    score_map = {
        (query, candidates[0].chunk.text): 3.0,
        (query, candidates[1].chunk.text): 2.0,
        (query, candidates[2].chunk.text): 1.0,
    }
    fake_model = FakeCrossEncoder(score_map=score_map)
    reranker = CrossEncoderReranker(model=fake_model)

    # top_k=2 should return top 2
    top2 = reranker.rerank(query=query, candidates=candidates, top_k=2)
    assert len(top2) == 2
    assert top2[0].chunk.chunk_id == "chunk_001"
    assert top2[1].chunk.chunk_id == "chunk_002"

    # top_k=None should return all 3
    all_res = reranker.rerank(query=query, candidates=candidates, top_k=None)
    assert len(all_res) == 3

    # top_k > len(candidates) returns all 3
    excess = reranker.rerank(query=query, candidates=candidates, top_k=50)
    assert len(excess) == 3

    # top_k <= 0 returns empty list
    assert reranker.rerank(query=query, candidates=candidates, top_k=0) == []
    assert reranker.rerank(query=query, candidates=candidates, top_k=-3) == []


def test_reranker_empty_inputs():
    """Verify empty candidates, empty query, and whitespace query return empty list."""
    candidates = _create_sample_fused_candidates()
    fake_model = FakeCrossEncoder(default_score=1.0)
    reranker = CrossEncoderReranker(model=fake_model)

    # Empty candidate list
    assert reranker.rerank(query="Valid query", candidates=[]) == []

    # Empty query string
    assert reranker.rerank(query="", candidates=candidates) == []

    # Whitespace-only query
    assert reranker.rerank(query="   \t\n  ", candidates=candidates) == []


def test_reranker_non_finite_scores_raises():
    """Verify that non-finite scores (NaN, Inf) raise ValueError."""
    query = "Query"
    candidates = _create_sample_fused_candidates()

    # NaN score
    nan_model = FakeCrossEncoder(
        score_map={(query, candidates[0].chunk.text): float("nan")}
    )
    reranker_nan = CrossEncoderReranker(model=nan_model)
    with pytest.raises(ValueError, match="non-finite relevance scores"):
        reranker_nan.rerank(query=query, candidates=candidates)

    # Inf score
    inf_model = FakeCrossEncoder(
        score_map={(query, candidates[0].chunk.text): float("inf")}
    )
    reranker_inf = CrossEncoderReranker(model=inf_model)
    with pytest.raises(ValueError, match="non-finite relevance scores"):
        reranker_inf.rerank(query=query, candidates=candidates)


def test_reranker_deterministic_tie_breaking():
    """Verify deterministic tie-breaking by chunk_id when relevance scores are identical."""
    query = "Tie test query"
    c_b = Chunk("chunk_b", "d1", "f.txt", 1, 0, "Text B", 0, 6)
    c_a = Chunk("chunk_a", "d1", "f.txt", 1, 1, "Text A", 7, 13)
    f_b = FusedResult(chunk=c_b, rrf_score=0.01, rank=1)
    f_a = FusedResult(chunk=c_a, rrf_score=0.01, rank=2)

    # Both chunks get identical score = 2.0
    fake_model = FakeCrossEncoder(default_score=2.0)
    reranker = CrossEncoderReranker(model=fake_model)

    # Pass in [f_b, f_a]
    results = reranker.rerank(query=query, candidates=[f_b, f_a])

    assert len(results) == 2
    assert results[0].reranker_score == results[1].reranker_score
    # Should be sorted alphabetically by chunk_id: chunk_a then chunk_b
    assert results[0].chunk.chunk_id == "chunk_a"
    assert results[0].rerank_rank == 1
    assert results[1].chunk.chunk_id == "chunk_b"
    assert results[1].rerank_rank == 2


def test_reranker_passes_correct_pairs_to_model():
    """Verify query-passage pairs passed to model.predict match (query, cand.chunk.text)."""
    query = "Precision pair test"
    candidates = _create_sample_fused_candidates()
    fake_model = FakeCrossEncoder(default_score=1.0)
    reranker = CrossEncoderReranker(model=fake_model)

    reranker.rerank(query=query, candidates=candidates)

    expected_pairs = [(query, c.chunk.text) for c in candidates]
    assert fake_model.recorded_pairs == expected_pairs


def test_reranker_batch_size_forwarding():
    """Verify batch_size configuration is forwarded to model.predict."""
    query = "Batch test"
    candidates = _create_sample_fused_candidates()
    fake_model = FakeCrossEncoder(default_score=1.0)
    reranker = CrossEncoderReranker(model=fake_model, batch_size=8)

    reranker.rerank(query=query, candidates=candidates)

    assert fake_model.recorded_batch_sizes == [8]


def test_reranker_direct_retrieval_result_candidates():
    """Verify reranker accepts direct RetrievalResult candidates without crashing."""
    query = "Direct retrieval test"
    c = Chunk("c1", "d1", "f.txt", 1, 0, "Direct text", 0, 11)
    retrieval_res = RetrievalResult(chunk=c, score=0.75, rank=1)

    fake_model = FakeCrossEncoder(default_score=3.5)
    reranker = CrossEncoderReranker(model=fake_model)

    results = reranker.rerank(query=query, candidates=[retrieval_res])

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].reranker_score == 3.5
    assert results[0].rerank_rank == 1
    assert results[0].rrf_score is None
    assert results[0].rrf_rank is None

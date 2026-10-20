"""Unit tests for Information Retrieval evaluation metrics.

100% offline, deterministic tests verifying Recall@k, Reciprocal Rank,
and Mean Reciprocal Rank (MRR) across standard, boundary, and edge cases.
"""

import pytest

from hybrid_rag.evaluation import (
    calculate_mrr,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)


def test_recall_at_k_standard_cases():
    """Verify Recall@k calculation with full, partial, and zero hits."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c2", "c4"}

    # k=1: contains "c1" (0 hits) -> Recall = 0.0
    assert calculate_recall_at_k(retrieved, relevant, k=1) == 0.0

    # k=2: contains "c1", "c2" (1 hit out of 2) -> Recall = 0.5
    assert calculate_recall_at_k(retrieved, relevant, k=2) == 0.5

    # k=3: contains "c1", "c2", "c3" (1 hit out of 2) -> Recall = 0.5
    assert calculate_recall_at_k(retrieved, relevant, k=3) == 0.5

    # k=4: contains "c1", "c2", "c3", "c4" (2 hits out of 2) -> Recall = 1.0
    assert calculate_recall_at_k(retrieved, relevant, k=4) == 1.0

    # k=10: exceeds retrieved length, all relevant retrieved -> Recall = 1.0
    assert calculate_recall_at_k(retrieved, relevant, k=10) == 1.0


def test_recall_at_k_edge_cases():
    """Verify Recall@k boundary conditions (empty sets, non-positive k)."""
    # Empty relevant set
    assert calculate_recall_at_k(["c1", "c2"], set(), k=3) == 0.0

    # Empty retrieved list
    assert calculate_recall_at_k([], {"c1"}, k=3) == 0.0

    # Non-positive k
    assert calculate_recall_at_k(["c1"], {"c1"}, k=0) == 0.0
    assert calculate_recall_at_k(["c1"], {"c1"}, k=-5) == 0.0

    # Unknown identifiers in retrieved list
    assert calculate_recall_at_k(["unknown_1", "unknown_2"], {"c1"}, k=2) == 0.0


def test_reciprocal_rank_calculation():
    """Verify Reciprocal Rank computation across rank positions."""
    # Rank 1 hit -> 1/1 = 1.0
    assert calculate_reciprocal_rank(["c1", "c2", "c3"], {"c1"}) == 1.0

    # Rank 2 hit -> 1/2 = 0.5
    assert calculate_reciprocal_rank(["c1", "c2", "c3"], {"c2"}) == 0.5

    # Rank 3 hit -> 1/3 ~ 0.3333
    assert pytest.approx(calculate_reciprocal_rank(["c1", "c2", "c3"], {"c3"}), abs=1e-4) == 1.0 / 3.0

    # Multiple relevant items: should take the FIRST relevant rank
    # Relevant = {"c2", "c3"}. First hit is at rank 2 -> RR = 0.5
    assert calculate_reciprocal_rank(["c1", "c2", "c3"], {"c2", "c3"}) == 0.5

    # No relevant item found
    assert calculate_reciprocal_rank(["c1", "c2", "c3"], {"c99"}) == 0.0

    # Empty inputs
    assert calculate_reciprocal_rank([], {"c1"}) == 0.0
    assert calculate_reciprocal_rank(["c1"], set()) == 0.0


def test_mrr_calculation_and_mismatch_validation():
    """Verify Mean Reciprocal Rank (MRR) averaging and query count validation."""
    # Query 1: Rank 1 hit (RR = 1.0)
    # Query 2: Rank 2 hit (RR = 0.5)
    # Query 3: Not retrieved (RR = 0.0)
    # Expected MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
    all_retrieved = [
        ["c1", "c2"],
        ["c3", "c4"],
        ["c5", "c6"],
    ]
    all_relevant = [
        {"c1"},
        {"c4"},
        {"c99"},
    ]

    mrr = calculate_mrr(all_retrieved, all_relevant)
    assert pytest.approx(mrr, abs=1e-4) == 0.5

    # Empty inputs return 0.0
    assert calculate_mrr([], []) == 0.0

    # Mismatched query count raises ValueError
    with pytest.raises(ValueError, match="Mismatched query counts"):
        calculate_mrr([["c1"]], [{"c1"}, {"c2"}])

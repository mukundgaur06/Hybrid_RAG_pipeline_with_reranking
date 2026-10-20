"""Information Retrieval evaluation metrics for HybridRAG.

Provides mathematically rigorous, scale-invariant metrics for evaluating
retrieval and reranking quality against ground-truth relevance judgments:
- Recall@k: Proportion of ground-truth relevant documents found in the top k.
- Reciprocal Rank (RR): Reciprocal of the 1-based rank position of the first relevant document.
- Mean Reciprocal Rank (MRR): Mean RR across a set of evaluation queries.
"""

from typing import Dict, List, Sequence, Set, Union


def calculate_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Calculate Recall@k for a single query.

    Formula:
        Recall@k = |Retrieved[:k] ∩ Relevant| / |Relevant|

    Args:
        retrieved_ids: Ordered sequence of retrieved chunk identifiers.
        relevant_ids: Set of ground-truth relevant chunk identifiers.
        k: Cut-off rank (positive integer).

    Returns:
        Float value in [0.0, 1.0]. If relevant_ids is empty or k <= 0, returns 0.0.
    """
    if not relevant_ids or k <= 0 or not retrieved_ids:
        return 0.0

    top_k_retrieved = set(retrieved_ids[:k])
    hits = len(top_k_retrieved.intersection(relevant_ids))
    return float(hits) / float(len(relevant_ids))


def calculate_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    Formula:
        RR = 1 / rank_first_relevant (where rank is 1-based)
        RR = 0.0 if no relevant item is found in retrieved_ids.

    Args:
        retrieved_ids: Ordered sequence of retrieved chunk identifiers.
        relevant_ids: Set of ground-truth relevant chunk identifiers.

    Returns:
        Float value in [0.0, 1.0].
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / float(rank)

    return 0.0


def calculate_mrr(
    all_retrieved_ids: Sequence[Sequence[str]],
    all_relevant_ids: Sequence[Set[str]],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across all queries.

    Formula:
        MRR = (1 / |Q|) * sum_{q in Q} RR(q)

    Args:
        all_retrieved_ids: List of retrieved ID lists, one per query.
        all_relevant_ids: List of relevant ID sets, one per query.

    Returns:
        Float value in [0.0, 1.0]. Returns 0.0 if queries list is empty.
    """
    if not all_retrieved_ids or not all_relevant_ids:
        return 0.0

    if len(all_retrieved_ids) != len(all_relevant_ids):
        raise ValueError(
            f"Mismatched query counts: {len(all_retrieved_ids)} retrieved lists "
            f"vs {len(all_relevant_ids)} relevant sets."
        )

    rr_scores = [
        calculate_reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved_ids, all_relevant_ids)
    ]

    return float(sum(rr_scores)) / float(len(rr_scores))

"""Cross-Encoder precision reranker using Sentence Transformers.

Takes a candidate pool of retrieved/fused chunks and performs full cross-attention
over (query, chunk_text) pairs to generate precise semantic relevance scores.
"""

from typing import Any, List, Optional, Sequence, Union

import numpy as np

from hybrid_rag.schema import FusedResult, RerankedResult, RetrievalResult


class CrossEncoderReranker:
    """Second-stage precision reranker powered by a Cross-Encoder model.

    Unlike Bi-Encoders that embed queries and documents into separate vector spaces,
    a Cross-Encoder takes the query and document concatenated together into self-attention
    layers, capturing cross-sentence token interactions with high precision.
    """

    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        model: Optional[Any] = None,
        batch_size: int = 16,
    ):
        """Initialize the Cross-Encoder reranker.

        Args:
            model_name_or_path: HuggingFace model identifier or local directory.
            model: Optional pre-instantiated CrossEncoder instance (or test double).
                   If None, the model is lazily loaded upon the first call to rerank().
            batch_size: Inference batch size for predicting candidate pair scores.
        """
        self.model_name_or_path = model_name_or_path
        self._model = model
        self.batch_size = max(1, batch_size)

    def _get_model(self) -> Any:
        """Lazily instantiate the CrossEncoder model if not injected."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name_or_path)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[Union[FusedResult, RetrievalResult]],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Score and re-rank a candidate pool against the query.

        Args:
            query: The user query string.
            candidates: Candidate items to re-rank (typically FusedResult or RetrievalResult).
            top_k: Optional maximum number of top reranked results to return.
                   If None, all input candidates are returned in reranked order.

        Returns:
            A list of RerankedResult objects ordered by reranker_score descending.

        Raises:
            ValueError: If the model produces non-finite scores (NaN or Inf).
        """
        if not candidates or not query or not query.strip():
            return []

        if top_k is not None and top_k <= 0:
            return []

        # Form (query, document_text) pairs for joint cross-attention
        pairs = [(query, cand.chunk.text) for cand in candidates]

        # Model inference
        model = self._get_model()
        raw_scores = model.predict(pairs, batch_size=self.batch_size)
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)

        # Validate numerical stability
        if not np.all(np.isfinite(scores)):
            raise ValueError("Cross-Encoder produced non-finite relevance scores (NaN or Inf).")

        # Associate each candidate with its score and chunk_id for tie-breaking
        scored_candidates = [
            (float(scores[idx]), candidates[idx].chunk.chunk_id, candidates[idx])
            for idx in range(len(candidates))
        ]

        # Deterministic sort: highest relevance score first; tie-break by chunk_id ascending
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        # Limit to top_k if specified
        effective_k = len(scored_candidates) if top_k is None else min(top_k, len(scored_candidates))
        selected_candidates = scored_candidates[:effective_k]

        # Construct 1-based ranked RerankedResult objects with complete source provenance
        results: List[RerankedResult] = []
        for rank, (score_val, _, cand) in enumerate(selected_candidates, start=1):
            # Extract metadata depending on whether candidate is FusedResult or RetrievalResult
            rrf_score = getattr(cand, "rrf_score", None)
            rrf_rank = getattr(cand, "rank", None) if isinstance(cand, FusedResult) else None
            bm25_rank = getattr(cand, "bm25_rank", None)
            bm25_score = getattr(cand, "bm25_score", None)
            dense_rank = getattr(cand, "dense_rank", None)
            dense_score = getattr(cand, "dense_score", None)

            # If candidate came directly from BM25 or Dense retriever without fusion
            if isinstance(cand, RetrievalResult):
                # We can deduce whether it was BM25 or Dense based on score magnitude if needed,
                # but preserving exact fields cleanly handles direct input
                pass

            results.append(
                RerankedResult(
                    chunk=cand.chunk,
                    reranker_score=score_val,
                    rerank_rank=rank,
                    rrf_score=rrf_score,
                    rrf_rank=rrf_rank,
                    bm25_rank=bm25_rank,
                    bm25_score=bm25_score,
                    dense_rank=dense_rank,
                    dense_score=dense_score,
                )
            )

        return results

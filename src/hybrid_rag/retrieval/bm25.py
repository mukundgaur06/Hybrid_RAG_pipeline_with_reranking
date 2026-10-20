"""BM25 lexical retrieval engine using rank-bm25.

Provides consistent tokenization for documents and queries, builds an in-memory
inverted index over Chunk objects, and returns ranked candidates with raw scores
and 1-based ranks.
"""

import re
from typing import List, Optional

from rank_bm25 import BM25Okapi

from hybrid_rag.schema import Chunk, RetrievalResult


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms for lexical indexing and querying.

    Using the exact same tokenization function for both document indexing and query
    processing ensures that vocabulary representations are consistent and reproducible.

    Args:
        text: Input string to tokenize.

    Returns:
        List of lowercase word tokens.
    """
    if not text:
        return []
    # Match contiguous word characters (letters, numbers, underscores)
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """In-memory lexical retriever based on BM25Okapi."""

    def __init__(self, chunks: Optional[List[Chunk]] = None):
        """Initialize and optionally index an initial list of Chunks.

        Args:
            chunks: Optional list of Chunk objects to index immediately.
        """
        self._chunks: List[Chunk] = []
        self._corpus_tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

        if chunks:
            self.index_chunks(chunks)

    @property
    def total_chunks(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build or replace the in-memory BM25 index with the provided Chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        self._chunks = list(chunks)
        self._corpus_tokens = [tokenize(c.text) for c in self._chunks]

        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_unmatched: bool = True,
    ) -> List[RetrievalResult]:
        """Retrieve the top-K chunks most lexically relevant to the query.

        Args:
            query: The user query string.
            top_k: Maximum number of ranked results to return.
            filter_unmatched: If True, exclude chunks with a BM25 score <= 0.0
                              (i.e., chunks containing no matching query terms).

        Returns:
            A list of RetrievalResult objects ordered by score descending.
        """
        if top_k <= 0 or not self._chunks or self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)

        # Compute raw BM25 scores across the entire corpus
        raw_scores = self._bm25.get_scores(query_tokens)

        # Pair each chunk with its score and index
        scored_candidates = []
        for idx, (chunk, score, doc_tokens) in enumerate(
            zip(self._chunks, raw_scores, self._corpus_tokens)
        ):
            score_val = float(score)
            if filter_unmatched and not any(t in query_token_set for t in doc_tokens):
                continue
            scored_candidates.append((score_val, chunk.chunk_id, idx, chunk))

        # Deterministic sorting: highest score first; tie-break by chunk_id ascending
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        # Limit to top_k
        top_candidates = scored_candidates[:top_k]

        # Construct 1-based ranked RetrievalResult objects
        results: List[RetrievalResult] = []
        for rank, (score_val, _, _, chunk) in enumerate(top_candidates, start=1):
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score_val,
                    rank=rank,
                )
            )

        return results

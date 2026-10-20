"""Dense semantic retrieval engine using Sentence Transformers and NumPy cosine similarity.

Encodes document chunks once during index creation into normalized dense vectors,
and computes inner-product cosine similarity against query embeddings at runtime.
"""

from typing import Any, List, Optional

import numpy as np

from hybrid_rag.schema import Chunk, RetrievalResult


class DenseRetriever:
    """In-memory semantic vector retriever using dense embeddings.

    Supports dependency injection of a custom or fake embedding model to enable
    deterministic, 100% offline unit testing without external network access.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        model: Optional[Any] = None,
        chunks: Optional[List[Chunk]] = None,
    ):
        """Initialize the dense retriever.

        Args:
            model_name_or_path: HuggingFace identifier or local path for the bi-encoder model.
            model: Optional pre-instantiated embedding model (or test double). If None,
                   the model will be lazily loaded from model_name_or_path on first use.
            chunks: Optional initial list of Chunk objects to index.
        """
        self.model_name_or_path = model_name_or_path
        self._model = model
        self._chunks: List[Chunk] = []
        self._normalized_embeddings: Optional[np.ndarray] = None

        if chunks:
            self.index_chunks(chunks)

    @property
    def total_chunks(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def _get_model(self) -> Any:
        """Lazily load the SentenceTransformer model if not already instantiated."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name_or_path)
        return self._model

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into a 2D float32 NumPy array."""
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True)
        arr = np.asarray(vectors, dtype=np.float32)

        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        if not np.all(np.isfinite(arr)):
            raise ValueError("Generated embeddings contain non-finite values (NaN or Inf).")

        return arr

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """Normalize row vectors to unit Euclidean (L2) norm for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero on degenerate zero-vectors
        norms = np.where(norms == 0, 1e-12, norms)
        return vectors / norms

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Encode all chunks and store their normalized vectors in an in-memory matrix.

        Document chunks are encoded exactly once at index time, ensuring query
        latency remains lightweight during inference.

        Args:
            chunks: List of Chunk objects to encode and index.
        """
        if not chunks:
            self._chunks = []
            self._normalized_embeddings = None
            return

        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]
        raw_vectors = self._encode_texts(texts)
        self._normalized_embeddings = self._normalize(raw_vectors)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve top-K chunks most semantically similar to the query using cosine similarity.

        Args:
            query: The user query string.
            top_k: Maximum number of ranked candidates to return.

        Returns:
            A list of RetrievalResult objects ordered by cosine similarity descending.
        """
        if (
            top_k <= 0
            or not self._chunks
            or self._normalized_embeddings is None
            or not query
            or not query.strip()
        ):
            return []

        # Bound top_k to the number of available chunks
        effective_k = min(top_k, len(self._chunks))

        # Encode and normalize query vector
        query_vector = self._encode_texts([query])
        normalized_query = self._normalize(query_vector)[0]  # shape (d,)

        # Compute cosine similarity via matrix-vector inner product: E_norm . q_norm
        # Since both matrix rows and query vector are unit-normalized, this is exact cosine similarity
        similarities = np.dot(self._normalized_embeddings, normalized_query)

        # Clip values to [-1.0, 1.0] for floating-point precision safety
        similarities = np.clip(similarities, -1.0, 1.0)

        # Pair scores with chunks and sort deterministically
        scored_candidates = [
            (float(similarities[idx]), self._chunks[idx].chunk_id, self._chunks[idx])
            for idx in range(len(self._chunks))
        ]

        # Sort descending by similarity score; tie-break by chunk_id ascending
        scored_candidates.sort(key=lambda item: (-item[0], item[1]))

        # Limit to top_k and build 1-based ranked RetrievalResult objects
        results: List[RetrievalResult] = []
        for rank, (score_val, _, chunk) in enumerate(scored_candidates[:effective_k], start=1):
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score_val,
                    rank=rank,
                )
            )

        return results

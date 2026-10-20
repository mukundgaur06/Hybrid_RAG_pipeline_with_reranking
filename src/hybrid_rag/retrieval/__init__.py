"""Retrieval subpackage for HybridRAG.

Contains:
- BM25Retriever: Lexical term-frequency/inverse-document-frequency retrieval
- DenseRetriever: Semantic bi-encoder embedding retrieval using NumPy cosine similarity
- reciprocal_rank_fusion: Scale-invariant rank fusion combining candidate lists
"""

from hybrid_rag.retrieval.bm25 import BM25Retriever, tokenize
from hybrid_rag.retrieval.dense import DenseRetriever
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "tokenize",
    "DenseRetriever",
    "reciprocal_rank_fusion",
]

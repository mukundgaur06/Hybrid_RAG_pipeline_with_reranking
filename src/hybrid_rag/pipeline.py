"""Integrated HybridRAG pipeline orchestrating ingestion, retrieval, reranking, and generation.

Connects DocumentLoader, TextChunker, BM25Retriever, DenseRetriever,
reciprocal_rank_fusion, CrossEncoderReranker, and GroundedAnswerGenerator
into a unified, robust query-answering system.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Union

from hybrid_rag.chunker import TextChunker
from hybrid_rag.generator import GroundedAnswerGenerator
from hybrid_rag.loader import DocumentLoader
from hybrid_rag.reranker import CrossEncoderReranker
from hybrid_rag.retrieval import BM25Retriever, DenseRetriever, reciprocal_rank_fusion
from hybrid_rag.schema import Chunk, PipelineResponse


class HybridRAGPipeline:
    """Unified pipeline orchestrating document ingestion and grounded hybrid RAG answering."""

    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[TextChunker] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        generator: Optional[GroundedAnswerGenerator] = None,
        candidate_pool_size: int = 6,
        final_context_size: int = 3,
        rrf_k: int = 60,
    ):
        """Initialize the HybridRAG pipeline.

        Args:
            loader: Optional DocumentLoader instance.
            chunker: Optional TextChunker instance.
            bm25_retriever: Optional pre-configured BM25Retriever.
            dense_retriever: Optional pre-configured DenseRetriever.
            reranker: Optional CrossEncoderReranker instance.
            generator: Optional GroundedAnswerGenerator instance.
            candidate_pool_size: Number of top candidates to retrieve from BM25 and Dense
                                before RRF fusion and reranking.
            final_context_size: Number of top reranked chunks to pass as evidence to the generator.
            rrf_k: Reciprocal Rank Fusion smoothing constant (default: 60).
        """
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or TextChunker(chunk_size=500, chunk_overlap=100)
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.generator = generator or GroundedAnswerGenerator()

        self.candidate_pool_size = max(1, candidate_pool_size)
        self.final_context_size = max(1, final_context_size)
        self.rrf_k = max(1, rrf_k)

        self._chunks: List[Chunk] = []

    @property
    def total_chunks(self) -> int:
        """Total number of indexed chunks currently in memory."""
        return len(self._chunks)

    def ingest_documents(self, paths: Sequence[Union[str, Path]]) -> int:
        """Ingest, chunk, and index documents in memory once.

        Subsequent queries against the corpus reuse these indexes without re-embedding.

        Args:
            paths: Sequence of file paths (.pdf, .txt, .md) to load.

        Returns:
            Total count of indexed chunks.
        """
        all_chunks: List[Chunk] = []
        for p in paths:
            doc = self.loader.load(p)
            doc_chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        self._chunks = all_chunks

        if all_chunks:
            # Build indexes once in memory
            if self.bm25_retriever is None:
                self.bm25_retriever = BM25Retriever(chunks=all_chunks)
            else:
                self.bm25_retriever.index_chunks(all_chunks)

            if self.dense_retriever is None:
                self.dense_retriever = DenseRetriever(chunks=all_chunks)
            else:
                self.dense_retriever.index_chunks(all_chunks)

        return len(self._chunks)

    def query(self, question: str) -> PipelineResponse:
        """Execute the multi-stage hybrid RAG pipeline for a given question.

        Stages:
        1. Lexical retrieval (BM25) and Dense semantic retrieval (Bi-Encoder)
        2. Reciprocal Rank Fusion (RRF) to merge candidate lists
        3. Cross-Encoder precision reranking over the top candidate pool
        4. Grounded answer generation and citation attribution (or fallback)

        Args:
            question: User inquiry.

        Returns:
            A structured PipelineResponse containing answer, sources, citations, and provenance.
        """
        # Guard: invalid question
        if not question or not question.strip():
            return PipelineResponse(
                query=question or "",
                answer="No query was provided.",
                sources=[],
                is_fallback=False,
            )

        # Guard: empty corpus
        if not self._chunks:
            return PipelineResponse(
                query=question,
                answer="The document index is empty. Ingest documents before querying.",
                sources=[],
                is_fallback=False,
            )

        # Ensure retrievers exist
        if self.bm25_retriever is None or self.dense_retriever is None:
            return PipelineResponse(
                query=question,
                answer="Retrieval indexes are not initialized.",
                sources=[],
                is_fallback=False,
            )

        # Stage 1: Parallel candidate generation
        bm25_results = self.bm25_retriever.retrieve(
            question, top_k=self.candidate_pool_size
        )
        dense_results = self.dense_retriever.retrieve(
            question, top_k=self.candidate_pool_size
        )

        # Stage 1: Reciprocal Rank Fusion
        fused_candidates = reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            k=self.rrf_k,
            top_k=self.candidate_pool_size,
        )

        if not fused_candidates:
            return PipelineResponse(
                query=question,
                answer="No relevant documents or evidence found to answer this question.",
                sources=[],
                is_fallback=False,
            )

        # Stage 2: Cross-Encoder precision reranking
        reranked_candidates = self.reranker.rerank(
            query=question,
            candidates=fused_candidates,
            top_k=self.final_context_size,
        )

        # Stage 3: Grounded answer generation
        return self.generator.generate(query=question, candidates=reranked_candidates)

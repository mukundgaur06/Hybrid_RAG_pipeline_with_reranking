"""Data schemas for HybridRAG documents and chunks.

Using typed dataclasses ensures that all components across the pipeline
(loaders, chunkers, retrievers, rerankers, generators) operate on predictable,
well-documented data structures with explicit metadata provenance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentPage:
    """Represents the text and page metadata of a single document page.

    Attributes:
        page_number: 1-based page index for paginated documents (like PDF),
                     or None for unpaginated text files.
        text: The raw textual content extracted from this page.
    """
    page_number: Optional[int]
    text: str


@dataclass
class Document:
    """Represents a fully loaded document with its constituent pages and metadata.

    Attributes:
        document_id: Deterministic unique identifier (e.g. SHA-256 hash or slug).
        filename: Name of the original file (e.g. "manual.pdf").
        file_type: Canonical file extension (e.g. "pdf", "txt", "md").
        pages: List of DocumentPage instances comprising the document.
        metadata: Arbitrary document-level metadata (file size, page count, etc.).
    """
    document_id: str
    filename: str
    file_type: str
    pages: List[DocumentPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenates all pages into a single string with double newlines."""
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def total_pages(self) -> int:
        """Returns the total number of pages in the document."""
        return len(self.pages)


@dataclass
class Chunk:
    """Represents a discrete text segment derived from a document for indexing.

    Attributes:
        chunk_id: Globally unique identifier for this chunk (e.g. "doc1_p1_c0").
        document_id: Unique identifier of the parent document.
        document_name: Filename of the source document for citation displays.
        page_number: 1-based page number where the chunk originated, or None.
        chunk_index: 0-based sequential index of this chunk within the document.
        text: The raw text content of the chunk.
        char_start: 0-based character start offset relative to the page/document text.
        char_end: 0-based character end offset relative to the page/document text.
        metadata: Additional context (e.g. chunk size, overlap setting, token estimate).
    """
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Represents a single retrieved chunk from a single retrieval engine.

    Attributes:
        chunk: The underlying Chunk object preserving full text and metadata.
        score: The raw score assigned by the retriever (e.g. BM25 score or cosine similarity).
        rank: 1-based rank position within this retriever's candidate list.
    """
    chunk: Chunk
    score: float
    rank: int


@dataclass
class FusedResult:
    """Represents a candidate chunk resulting from rank-based fusion (e.g. RRF).

    Attributes:
        chunk: The underlying Chunk object.
        rrf_score: The calculated Reciprocal Rank Fusion score.
        rank: 1-based rank position within the fused candidate list.
        bm25_rank: Original 1-based rank from BM25 retrieval, or None if not present.
        bm25_score: Original raw score from BM25 retrieval, or None if not present.
        dense_rank: Original 1-based rank from Dense retrieval, or None if not present.
        dense_score: Original cosine similarity from Dense retrieval, or None if not present.
    """
    chunk: Chunk
    rrf_score: float
    rank: int
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None


@dataclass
class RerankedResult:
    """Represents a candidate chunk scored and re-ordered by the Cross-Encoder reranker.

    Attributes:
        chunk: The underlying Chunk object preserving full text and metadata.
        reranker_score: The raw relevance logit assigned by the Cross-Encoder.
        rerank_rank: 1-based rank position after Cross-Encoder precision reranking.
        rrf_score: Original candidate score from Reciprocal Rank Fusion, if present.
        rrf_rank: Original 1-based rank position from Reciprocal Rank Fusion, if present.
        bm25_rank: Original 1-based rank from BM25 lexical retrieval, if present.
        bm25_score: Original raw score from BM25 lexical retrieval, if present.
        dense_rank: Original 1-based rank from Dense vector retrieval, if present.
        dense_score: Original cosine similarity from Dense vector retrieval, if present.
    """
    chunk: Chunk
    reranker_score: float
    rerank_rank: int
    rrf_score: Optional[float] = None
    rrf_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None


@dataclass
class SourcePassage:
    """Represents a passage formatted as an explicit evidence source for the generator.

    Attributes:
        source_id: Explicit citation identifier, e.g. '[S1]', '[S2]'.
        chunk: The underlying Chunk containing text, document name, page, and offsets.
        reranked_result: The full RerankedResult preserving retrieval and fusion provenance.
    """
    source_id: str
    chunk: Chunk
    reranked_result: Optional[RerankedResult] = None


@dataclass
class PipelineResponse:
    """Structured response returned by the HybridRAG pipeline and generator.

    Attributes:
        query: The user's query string.
        answer: Synthesized answer text, or descriptive evidence status when in fallback mode.
        sources: Ordered list of assigned evidence passages.
        citations: List of validated citation IDs (e.g. ['[S1]', '[S2]']) referenced in the answer.
        unvalidated_citations: List of citation IDs cited in the answer that were not present in the sources.
        is_fallback: True if generation fell back to Evidence Viewer mode (e.g., missing API key or API error).
        fallback_reason: Explanation of why fallback mode was activated, if applicable.
        model_name: Name of the generation model used, or None if generation was skipped.
    """
    query: str
    answer: str
    sources: List[SourcePassage] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    unvalidated_citations: List[str] = field(default_factory=list)
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    model_name: Optional[str] = None


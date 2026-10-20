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

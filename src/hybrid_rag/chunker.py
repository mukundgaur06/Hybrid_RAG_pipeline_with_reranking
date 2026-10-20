"""Text chunking engine with metadata preservation and page-boundary integrity.

Splits document text into overlapping segments while tracking character offsets,
chunk indices, and source page provenance.
"""

from typing import List

from hybrid_rag.schema import Chunk, Document


class TextChunker:
    """Configurable sliding-window text chunker.

    Attributes:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """Initialize the chunker with size and overlap constraints.

        Args:
            chunk_size: Target character length for each chunk (must be > 0).
            chunk_overlap: Character overlap between consecutive chunks
                           (must be >= 0 and strictly < chunk_size).

        Raises:
            ValueError: If chunk_size or chunk_overlap violate constraints.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk all pages of a Document into a sequential list of Chunks.

        To guarantee strict page-level citation provenance, paginated documents
        (e.g. PDFs) are chunked on a per-page basis. This prevents chunks from
        crossing page boundaries and creating ambiguous citations.

        Args:
            document: The Document to be chunked.

        Returns:
            A list of Chunk objects with metadata and provenance.
        """
        chunks: List[Chunk] = []
        global_chunk_index = 0

        for page in document.pages:
            text = page.text
            if not text or not text.strip():
                continue

            text_len = len(text)
            start = 0

            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunk_slice = text[start:end]

                # Only emit chunks containing non-whitespace content
                if chunk_slice.strip():
                    # Construct unique, human-readable chunk_id
                    if page.page_number is not None:
                        chunk_id = f"{document.document_id}_p{page.page_number}_c{global_chunk_index}"
                    else:
                        chunk_id = f"{document.document_id}_c{global_chunk_index}"

                    chunk = Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        document_name=document.filename,
                        page_number=page.page_number,
                        chunk_index=global_chunk_index,
                        text=chunk_slice,
                        char_start=start,
                        char_end=end,
                        metadata={
                            "chunk_size_chars": len(chunk_slice),
                            "configured_chunk_size": self.chunk_size,
                            "configured_chunk_overlap": self.chunk_overlap,
                        },
                    )
                    chunks.append(chunk)
                    global_chunk_index += 1

                if end >= text_len:
                    break

                start += self.step_size

        return chunks

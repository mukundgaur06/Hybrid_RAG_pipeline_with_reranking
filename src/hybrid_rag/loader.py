"""Document loader supporting PDF, TXT, and Markdown files.

Extracts text content and preserves file and page-level metadata without
executing any file contents or relying on external services.
"""

import hashlib
from pathlib import Path
from typing import List, Union

try:
    import pymupdf  # PyMuPDF
except ImportError:
    import fitz as pymupdf  # Alternative import style for PyMuPDF

from hybrid_rag.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from hybrid_rag.schema import Document, DocumentPage


class DocumentLoader:
    """Loads and extracts text and metadata from PDF, TXT, and Markdown files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> Document:
        """Load a document from disk and extract its text and metadata.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            A Document object containing pages, metadata, and extracted text.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            UnsupportedFileTypeError: If the file extension is not supported.
            DocumentParsingError: If reading/parsing the file fails (e.g. corrupt PDF).
            EmptyDocumentError: If the file contains no readable text.
        """
        path = Path(file_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(cls.SUPPORTED_EXTENSIONS))}"
            )

        # Generate a deterministic document ID based on file path and size
        doc_id = cls._generate_document_id(path)
        filename = path.name

        try:
            if ext == ".pdf":
                pages = cls._load_pdf(path)
            elif ext in {".txt", ".md"}:
                pages = cls._load_text(path)
            else:
                raise UnsupportedFileTypeError(f"Unsupported extension: {ext}")
        except (UnsupportedFileTypeError, EmptyDocumentError, FileNotFoundError):
            raise
        except Exception as exc:
            raise DocumentParsingError(f"Failed to parse document '{filename}': {exc}") from exc

        # Check for empty content
        total_text = "".join(page.text.strip() for page in pages)
        if not total_text:
            raise EmptyDocumentError(f"Document '{filename}' contains no readable text.")

        metadata = {
            "source_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "total_pages": len(pages),
        }

        return Document(
            document_id=doc_id,
            filename=filename,
            file_type=ext.lstrip("."),
            pages=pages,
            metadata=metadata,
        )

    @classmethod
    def _load_pdf(cls, path: Path) -> List[DocumentPage]:
        """Extract pages from a PDF using PyMuPDF with 1-based page indexing."""
        pages: List[DocumentPage] = []
        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            raise DocumentParsingError(f"Corrupt or unreadable PDF '{path.name}': {exc}") from exc

        with doc:
            if doc.is_encrypted:
                raise DocumentParsingError(f"Encrypted/password-protected PDFs are not supported: '{path.name}'")

            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text()
                # 1-based page numbering for human readability and citation displays
                page_number = page_index + 1
                pages.append(DocumentPage(page_number=page_number, text=text))

        return pages

    @classmethod
    def _load_text(cls, path: Path) -> List[DocumentPage]:
        """Load plaintext or markdown files with UTF-8 encoding (and fallback)."""
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="latin-1")
            except Exception as exc:
                raise DocumentParsingError(f"Failed to decode text file '{path.name}': {exc}") from exc
        except Exception as exc:
            raise DocumentParsingError(f"Unable to read file '{path.name}': {exc}") from exc

        # Text and Markdown are unpaginated; treated as a single page with page_number=None
        return [DocumentPage(page_number=None, text=content)]

    @staticmethod
    def _generate_document_id(path: Path) -> str:
        """Create a deterministic, readable document identifier."""
        stat = path.stat()
        hasher = hashlib.sha256()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        short_hash = hasher.hexdigest()[:8]
        # Clean slug from filename
        clean_name = "".join(c if c.isalnum() else "_" for c in path.stem).strip("_")
        return f"{clean_name}_{short_hash}"

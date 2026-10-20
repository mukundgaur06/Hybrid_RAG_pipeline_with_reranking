"""Custom exceptions for the HybridRAG pipeline.

Centralizing custom exceptions provides clear, actionable feedback when
document loading, parsing, chunking, or retrieval encounters invalid states.
"""


class HybridRAGError(Exception):
    """Base exception for all errors raised by the HybridRAG package."""
    pass


class UnsupportedFileTypeError(HybridRAGError):
    """Raised when a file with an unsupported extension is passed to the loader."""
    pass


class EmptyDocumentError(HybridRAGError):
    """Raised when an input document contains no readable or extractable text."""
    pass


class DocumentParsingError(HybridRAGError):
    """Raised when an error occurs during parsing (e.g. corrupt PDF, encoding failure)."""
    pass

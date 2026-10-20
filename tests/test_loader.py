"""Unit tests for DocumentLoader.

Verifies document extraction across TXT, Markdown, and PDF formats,
ensuring robust metadata preservation and failure mode handling.
"""

from pathlib import Path
import pytest
import pymupdf

from hybrid_rag.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from hybrid_rag.loader import DocumentLoader


def test_load_txt_file(tmp_path: Path):
    """Test loading a plain UTF-8 text file."""
    sample_text = "This is a plain text document.\nIt contains multiple lines for testing."
    file_path = tmp_path / "sample.txt"
    file_path.write_text(sample_text, encoding="utf-8")

    doc = DocumentLoader.load(file_path)

    assert doc.filename == "sample.txt"
    assert doc.file_type == "txt"
    assert doc.total_pages == 1
    assert doc.pages[0].page_number is None
    assert "plain text document" in doc.full_text
    assert doc.metadata["file_size_bytes"] == file_path.stat().st_size


def test_load_markdown_file(tmp_path: Path):
    """Test loading a Markdown document."""
    sample_md = "# Architecture\n\n- Lexical\n- Dense\n- Hybrid RRF"
    file_path = tmp_path / "overview.md"
    file_path.write_text(sample_md, encoding="utf-8")

    doc = DocumentLoader.load(file_path)

    assert doc.filename == "overview.md"
    assert doc.file_type == "md"
    assert doc.total_pages == 1
    assert "Hybrid RRF" in doc.full_text


def test_load_pdf_file_multipage(tmp_path: Path):
    """Test creating and extracting a 2-page PDF with 1-based page numbers."""
    pdf_path = tmp_path / "multipage_doc.pdf"

    # Create synthetic 2-page PDF using PyMuPDF
    pdf = pymupdf.open()
    page1 = pdf.new_page()
    page1.insert_text((50, 72), "Content on page one.")
    page2 = pdf.new_page()
    page2.insert_text((50, 72), "Content on page two.")
    pdf.save(str(pdf_path))
    pdf.close()

    doc = DocumentLoader.load(pdf_path)

    assert doc.filename == "multipage_doc.pdf"
    assert doc.file_type == "pdf"
    assert doc.total_pages == 2
    assert doc.pages[0].page_number == 1
    assert "page one" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "page two" in doc.pages[1].text


def test_file_not_found():
    """Test that a non-existent file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        DocumentLoader.load("non_existent_document_12345.pdf")


def test_unsupported_file_extension(tmp_path: Path):
    """Test that unsupported formats (e.g. .docx, .csv) raise UnsupportedFileTypeError."""
    invalid_file = tmp_path / "data.docx"
    invalid_file.write_text("Dummy content", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        DocumentLoader.load(invalid_file)
    assert ".docx" in str(exc_info.value)


def test_empty_txt_file(tmp_path: Path):
    """Test that an empty or whitespace-only file raises EmptyDocumentError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\t  \n  ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        DocumentLoader.load(empty_file)


def test_empty_pdf_file(tmp_path: Path):
    """Test that a PDF with blank pages raises EmptyDocumentError."""
    pdf_path = tmp_path / "blank.pdf"
    pdf = pymupdf.open()
    pdf.new_page()  # Blank page without text
    pdf.save(str(pdf_path))
    pdf.close()

    with pytest.raises(EmptyDocumentError):
        DocumentLoader.load(pdf_path)


def test_corrupt_pdf_file(tmp_path: Path):
    """Test that a corrupted PDF file raises DocumentParsingError."""
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"NOT_A_VALID_PDF_STREAM_123456789")

    with pytest.raises(DocumentParsingError):
        DocumentLoader.load(corrupt_path)

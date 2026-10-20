"""Unit tests for TextChunker.

Verifies boundary calculations, exact overlap guarantees, metadata preservation,
per-page isolation for citations, and validation constraints.
"""

from pathlib import Path
import pytest
import pymupdf

from hybrid_rag.chunker import TextChunker
from hybrid_rag.loader import DocumentLoader
from hybrid_rag.schema import Document, DocumentPage


def test_chunk_short_text():
    """Text shorter than chunk_size should produce exactly one chunk matching the source slice."""
    source_text = "Short text snippet."
    doc = Document(
        document_id="test_doc_01",
        filename="notes.txt",
        file_type="txt",
        pages=[DocumentPage(page_number=None, text=source_text)],
    )
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == source_text
    assert c.chunk_index == 0
    assert c.page_number is None
    assert c.document_name == "notes.txt"
    assert c.chunk_id == "test_doc_01_c0"
    assert c.char_start == 0
    assert c.char_end == len(source_text)
    # Crucial invariant: source slice must strictly equal chunk.text
    assert doc.pages[0].text[c.char_start:c.char_end] == c.text


def test_chunk_boundaries_and_exact_overlap():
    """Verify that chunks advance by step_size and consecutive chunks share exact overlap text."""
    words = [f"{i:04d}" for i in range(200)]  # 200 * 5 = 1000 chars
    full_text = " ".join(words)

    doc = Document(
        document_id="test_doc_02",
        filename="stream.txt",
        file_type="txt",
        pages=[DocumentPage(page_number=None, text=full_text)],
    )

    chunk_size = 400
    chunk_overlap = 100
    step_size = chunk_size - chunk_overlap  # 300

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1

    # Verify every chunk strictly matches the source slice
    for c in chunks:
        assert doc.pages[0].text[c.char_start:c.char_end] == c.text
        assert len(c.text) == c.char_end - c.char_start

    # Verify consecutive chunk offsets and exact character-level overlap
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]

        assert c2.char_start == c1.char_start + step_size
        assert c1.char_end == c1.char_start + chunk_size

        # Exact overlap slice in source text
        expected_overlap = full_text[c2.char_start:c1.char_end]
        assert len(expected_overlap) == chunk_overlap
        # End of c1 must match overlap
        assert c1.text[step_size:] == expected_overlap
        # Start of c2 must match overlap
        assert c2.text[:chunk_overlap] == expected_overlap


def test_consecutive_chunk_overlap_on_natural_text():
    """Verify that chunk overlap behaves consistently on natural language with punctuation."""
    text = (
        "The Apollo program was an American human spaceflight initiative carried out by NASA. "
        "Its definitive accomplishment was landing the first humans on the Moon in July 1969. "
        "The Saturn V was a three-stage liquid-fueled super heavy-lift rocket."
    )
    doc = Document(
        document_id="apollo_test",
        filename="apollo.txt",
        file_type="txt",
        pages=[DocumentPage(page_number=None, text=text)],
    )

    chunk_size = 120
    chunk_overlap = 40
    step_size = chunk_size - chunk_overlap  # 80

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3

    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]

        # 1. Zero-based, half-open offset invariant
        assert 0 <= c1.char_start < c1.char_end <= len(text)
        assert text[c1.char_start:c1.char_end] == c1.text
        assert text[c2.char_start:c2.char_end] == c2.text

        # 2. Overlap invariant
        overlap_slice = text[c2.char_start:c1.char_end]
        assert len(overlap_slice) == chunk_overlap
        assert c1.text[-chunk_overlap:] == overlap_slice
        assert c2.text[:chunk_overlap] == overlap_slice


def test_chunk_multipage_pdf_preserves_page_numbers():
    """Test that chunks never cross page boundaries and preserve 1-based page numbers."""
    page1_text = "This is Page 1 content. " * 30  # ~720 chars
    page2_text = "This is Page 2 content. " * 30  # ~720 chars

    doc = Document(
        document_id="apollo_pdf_01",
        filename="apollo.pdf",
        file_type="pdf",
        pages=[
            DocumentPage(page_number=1, text=page1_text),
            DocumentPage(page_number=2, text=page2_text),
        ],
    )

    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(page1_chunks) > 0
    assert len(page2_chunks) > 0
    assert len(page1_chunks) + len(page2_chunks) == len(chunks)

    # Check that page 1 chunks ONLY contain page 1 text and slice correctly
    for c in page1_chunks:
        assert "Page 1" in c.text
        assert "Page 2" not in c.text
        assert "_p1_" in c.chunk_id
        assert doc.pages[0].text[c.char_start:c.char_end] == c.text

    # Check that page 2 chunks ONLY contain page 2 text and slice correctly
    for c in page2_chunks:
        assert "Page 2" in c.text
        assert "Page 1" not in c.text
        assert "_p2_" in c.chunk_id
        assert doc.pages[1].text[c.char_start:c.char_end] == c.text


def test_pdf_chunks_from_loaded_file(tmp_path: Path):
    """End-to-end test with DocumentLoader and TextChunker on a real multi-page PDF."""
    pdf_path = tmp_path / "sample_multipage.pdf"
    doc_pdf = pymupdf.open()

    p1 = doc_pdf.new_page()
    p1.insert_text((50, 72), "First page: Historical overview of lunar exploration.")

    p2 = doc_pdf.new_page()
    p2.insert_text((50, 72), "Second page: Technical specifications of the Saturn V rocket.")

    p3 = doc_pdf.new_page()
    p3.insert_text((50, 72), "Third page: Mission telemetry and splashdown logistics.")

    doc_pdf.save(str(pdf_path))
    doc_pdf.close()

    # Load and chunk
    document = DocumentLoader.load(pdf_path)
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(document)

    assert len(chunks) >= 3
    # Verify all chunks have 1-based page numbers matching their source page
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2, 3}

    # Verify chunks never cross page boundaries
    for c in chunks:
        assert c.page_number in {1, 2, 3}
        source_page = document.pages[c.page_number - 1]
        assert source_page.text[c.char_start:c.char_end] == c.text


def test_chunker_invalid_parameters():
    """Verify that invalid chunk_size or chunk_overlap values raise ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=-10, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        TextChunker(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_zero_overlap():
    """Verify chunking functions correctly with zero overlap."""
    text = "A" * 500
    doc = Document(
        document_id="zero_overlap_doc",
        filename="zero.txt",
        file_type="txt",
        pages=[DocumentPage(page_number=None, text=text)],
    )

    chunker = TextChunker(chunk_size=200, chunk_overlap=0)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 3
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == 200
    assert chunks[0].text == text[0:200]
    assert chunks[1].char_start == 200
    assert chunks[1].char_end == 400
    assert chunks[1].text == text[200:400]
    assert chunks[2].char_start == 400
    assert chunks[2].char_end == 500
    assert chunks[2].text == text[400:500]


def test_chunk_ids_are_unique():
    """Verify that all generated chunks have strictly unique IDs."""
    doc = Document(
        document_id="unique_test",
        filename="unique.txt",
        file_type="txt",
        pages=[
            DocumentPage(page_number=1, text="Hello world! " * 50),
            DocumentPage(page_number=2, text="Second page content. " * 50),
        ],
    )
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk IDs must be strictly unique"

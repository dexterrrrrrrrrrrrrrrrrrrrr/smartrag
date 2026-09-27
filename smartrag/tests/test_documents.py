"""Tests for document parsing and chunking."""
from pathlib import Path

import pytest

from backend.core.config import Settings
from backend.retrieval.chunker import chunk_document
from backend.retrieval.parsers import (
    DocumentParseError,
    UnsupportedFileTypeError,
    extract_pages,
)


def test_extract_pages_from_txt(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world. This is a test document.")
    pages = extract_pages(file_path)
    assert len(pages) == 1
    assert pages[0][1] is None  # no page number for plain text
    assert "Hello world" in pages[0][0]


def test_extract_pages_unsupported_type(tmp_path: Path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("irrelevant")
    with pytest.raises(UnsupportedFileTypeError):
        extract_pages(file_path)


def test_extract_pages_empty_file_raises(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   ")
    with pytest.raises(DocumentParseError):
        extract_pages(file_path)


def test_chunking_respects_configured_size():
    settings = Settings(chunk_size=50, chunk_overlap=10)
    text = "word " * 100  # 500 chars
    chunks = chunk_document("doc.txt", [(text, None)], source="doc.txt", settings=settings)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 60  # small slack for splitter boundary rounding


def test_chunking_preserves_page_numbers():
    settings = Settings(chunk_size=1000, chunk_overlap=0)
    pages = [("Page one content about RAG.", 1), ("Page two content about caching.", 2)]
    chunks = chunk_document("doc.pdf", pages, source="doc.pdf", settings=settings)
    page_numbers = {c.metadata.page_number for c in chunks}
    assert page_numbers == {1, 2}


def test_each_chunk_has_required_metadata_fields():
    settings = Settings(chunk_size=1000, chunk_overlap=0)
    chunks = chunk_document("doc.pdf", [("Some content here.", 3)], source="/path/doc.pdf", settings=settings)
    for c in chunks:
        assert c.metadata.document_name == "doc.pdf"
        assert c.metadata.page_number == 3
        assert c.metadata.chunk_id
        assert c.metadata.source == "/path/doc.pdf"
        assert c.metadata.created_at is not None

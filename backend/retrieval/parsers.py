"""
Text extraction for supported document types.

Each parser returns a list of (text, page_number) tuples so downstream
chunking can preserve page-level provenance for citations. TXT/Markdown
have no real pagination, so they're treated as a single "page" (None).
"""
from pathlib import Path

from pypdf import PdfReader

from backend.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class UnsupportedFileTypeError(Exception):
    def __init__(self, suffix: str):
        super().__init__(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )


class DocumentParseError(Exception):
    """Raised when a file exists and has a supported extension but its
    content could not be parsed (corrupt PDF, bad encoding, etc.)."""


def extract_pages(file_path: str | Path) -> list[tuple[str, int | None]]:
    """Return a list of (page_text, page_number) tuples for a document.
    page_number is 1-indexed for PDFs, None for plain text formats."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(suffix)

    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        else:  # .txt, .md, .markdown
            return _extract_plain_text(path)
    except (UnsupportedFileTypeError, DocumentParseError):
        raise
    except Exception as exc:  # noqa: BLE001 - convert any parser failure to a typed error
        raise DocumentParseError(f"Failed to parse '{path.name}': {exc}") from exc


def _extract_pdf(path: Path) -> list[tuple[str, int | None]]:
    reader = PdfReader(str(path))
    pages: list[tuple[str, int | None]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i))
    if not pages:
        raise DocumentParseError(
            f"'{path.name}' produced no extractable text (likely a scanned/image-only PDF)."
        )
    return pages


def _extract_plain_text(path: Path) -> list[tuple[str, int | None]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise DocumentParseError(f"'{path.name}' is empty.")
    return [(text, None)]

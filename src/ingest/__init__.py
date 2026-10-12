"""PDF ingestion package (Person 1 scope, implemented in P2 repo Phase 9L)."""

from src.ingest.pdf_ingest import (
    MAX_PDF_BYTES,
    PdfIngestError,
    is_pdf_bytes,
    parse_pdf_bytes,
)

__all__ = [
    "MAX_PDF_BYTES",
    "PdfIngestError",
    "is_pdf_bytes",
    "parse_pdf_bytes",
]

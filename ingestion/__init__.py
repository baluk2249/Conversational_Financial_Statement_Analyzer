"""Document parsing and conservative financial-fact normalization."""

from ingestion.parsers import (
    DocumentProcessingError,
    IngestionResult,
    UnsupportedDocumentType,
    parse_document,
)
from ingestion.pdf_tables import (
    ExtractedTable,
    PDFTableExtractionError,
    PDFTableExtractor,
    detect_statement_type,
)

__all__ = [
    "DocumentProcessingError",
    "ExtractedTable",
    "IngestionResult",
    "PDFTableExtractionError",
    "PDFTableExtractor",
    "UnsupportedDocumentType",
    "detect_statement_type",
    "parse_document",
]

from __future__ import annotations

from io import BytesIO

import pdfplumber
import pytest
from pypdf import PdfWriter

from application.service import AnalyzerService
from ingestion import ExtractedTable, PDFTableExtractor, detect_statement_type
from ingestion import parsers
from models.schemas import FinancialFact
from retrieval.structured_store import SQLiteFinancialStore


class InMemoryNarrativeStore:
    def __init__(self) -> None:
        self.chunks = []

    def add_chunks(self, chunks) -> None:
        self.chunks.extend(chunks)

    def search(
        self,
        query: str,
        company_id: str,
        period: str | None = None,
        document_type: str | None = None,
        top_k: int = 5,
    ):
        del query, company_id, period, document_type, top_k
        return []


def _blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    content = BytesIO()
    writer.write(content)
    return content.getvalue()


def test_statement_detector_recognizes_heading_variants() -> None:
    variants = {
        "income_statement": (
            "Income Statement",
            "Statement of Profit and Loss",
            "Profit and Loss Account",
            "Statement of Operations",
            "Consolidated Statement of Income",
        ),
        "balance_sheet": (
            "Balance Sheet",
            "Statement of Financial Position",
            "Consolidated Balance Sheet",
        ),
        "cash_flow_statement": (
            "Cash Flow Statement",
            "Statement of Cash Flows",
            "Consolidated Statement of Cash Flows",
        ),
    }
    for expected, headings in variants.items():
        assert all(detect_statement_type(heading) == expected for heading in headings)
    assert detect_statement_type("Quarterly customer satisfaction scores 2025") == "other"


def test_pdf_table_extractor_preserves_page_table_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        width = 612
        height = 792

        def extract_text(self) -> str:
            return "Statement of Financial Position"

        def extract_words(self):
            return []

        def extract_tables(self, table_settings=None):
            if table_settings is None:
                return []
            return [[
                ["Line item", "FY2025", "FY2024"],
                ["Current assets", "1,200", "1,000"],
                ["Current liabilities", "800", "700"],
            ]]

    class FakeDocument:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            del exception_type, exception, traceback
            return None

    monkeypatch.setattr(pdfplumber, "open", lambda _source: FakeDocument())
    tables = PDFTableExtractor().extract_tables(
        BytesIO(b"pdf"), source_file="annual-report.pdf"
    )

    assert len(tables) == 1
    assert tables[0].source_file == "annual-report.pdf"
    assert tables[0].page_number == 1
    assert tables[0].table_index == 1
    assert tables[0].headers == ["Line item", "FY2025", "FY2024"]
    assert tables[0].rows[0] == ["Current assets", "1,200", "1,000"]
    assert "Statement of Financial Position" in tables[0].extraction_metadata["page_text"]


def test_pdf_table_extractor_recovers_aligned_columns_when_grid_detection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        width = 612
        height = 792

        def extract_text(self) -> str:
            return "Statement of Financial Position"

        def extract_tables(self, table_settings=None):
            del table_settings
            return []

        def extract_words(self):
            rows = [
                [("Line", 50, 72.7), ("item", 76, 98.7), ("2025", 155.4, 182.1), ("2024", 225.5, 252.1)],
                [("Current", 50, 90), ("assets", 93.3, 128), ("1,200", 171.4, 201.4), ("1,000", 241.4, 271.5)],
                [("Current", 50, 90), ("liabilities", 93.3, 138.7), ("800", 172, 192.1), ("700", 238.8, 258.8)],
            ]
            return [
                {"text": text, "x0": left, "x1": right, "top": row_index * 25}
                for row_index, row in enumerate(rows)
                for text, left, right in row
            ]

    class FakeDocument:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            del exception_type, exception, traceback
            return None

    monkeypatch.setattr(pdfplumber, "open", lambda _source: FakeDocument())
    tables = PDFTableExtractor().extract_tables(BytesIO(b"pdf"))

    assert len(tables) == 1
    assert tables[0].headers == ["Line item", "2025", "2024"]
    assert tables[0].rows == [
        ["Current assets", "1,200", "1,000"],
        ["Current liabilities", "800", "700"],
    ]
    assert tables[0].extraction_metadata["extraction_method"] == "word_alignment"


def test_pdf_upload_produces_narrative_chunks_and_normalized_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = ExtractedTable(
        source_file="annual-report.pdf",
        page_number=12,
        table_index=1,
        headers=["Line item", "Year ended 31 December 2025", "Year ended 31 December 2024"],
        rows=[
            ["Revenue", "1,250", "1,100"],
            ["Operating profit", "250", "200"],
            ["Other numeric table row", "999", "888"],
        ],
        raw_text=(
            "Line item\tYear ended 31 December 2025\tYear ended 31 December 2024\n"
            "Revenue\t1,250\t1,100\nOperating profit\t250\t200\n"
        ),
        extraction_metadata={
            "page_text": (
                "Consolidated Statement of Income\n"
                "Amounts in £ millions\n"
                "Management discussed improved demand."
            )
        },
    )

    class FakePDFTableExtractor:
        def extract_tables(self, file_path, *, source_file=None):
            del file_path
            assert source_file == "annual-report.pdf"
            return [table]

    class FakePage:
        def extract_text(self) -> str:
            return (
                "Consolidated Statement of Income\n"
                "Amounts in £ millions\n"
                "Management discussed improved demand."
            )

    class FakeReader:
        def __init__(self, _source) -> None:
            self.pages = [FakePage()]

    monkeypatch.setattr(parsers, "PDFTableExtractor", FakePDFTableExtractor)
    monkeypatch.setattr("pypdf.PdfReader", FakeReader)

    narrative_store = InMemoryNarrativeStore()
    structured_store = SQLiteFinancialStore(":memory:")
    service = AnalyzerService(
        structured_store,
        narrative_store,
    )
    result = service.ingest_document(
        "annual-report.pdf",
        _blank_pdf(),
        company_name="Example plc",
        period="FY2025",
    )

    assert result["facts_stored"] == 4
    assert result["narrative_chunks"] == 1
    assert result["periods"] == ["FY2024", "FY2025"]
    assert result["statement_types"] == ["income_statement"]
    assert len(narrative_store.chunks) == 1
    assert narrative_store.chunks[0].page == 1

    current_revenue = structured_store.get_fact(
        "example-plc", "revenue", "FY2025", "income_statement"
    )
    prior_revenue = structured_store.get_fact(
        "example-plc", "revenue", "FY2024", "income_statement"
    )
    assert isinstance(current_revenue, FinancialFact)
    assert current_revenue.value == 1250
    assert current_revenue.currency == "GBP"
    assert current_revenue.unit == "millions"
    assert current_revenue.locator.page == 12
    assert current_revenue.locator.details["table_index"] == 1
    assert current_revenue.source_reference.citation == (
        "[Income Statement | FY2025 | Revenue | p.12]"
    )
    assert isinstance(prior_revenue, FinancialFact)
    assert prior_revenue.value == 1100


def test_unheaded_numeric_pdf_table_is_not_classified_as_a_statement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = ExtractedTable(
        source_file="annual-report.pdf",
        page_number=3,
        table_index=1,
        headers=["Category", "2025"],
        rows=[["Current assets", "120"], ["Current liabilities", "80"]],
        raw_text="Category\t2025\nCurrent assets\t120\nCurrent liabilities\t80",
        extraction_metadata={"page_text": "Regional revenue by market"},
    )

    class FakePDFTableExtractor:
        def extract_tables(self, file_path, *, source_file=None):
            del file_path, source_file
            return [table]

    class FakePage:
        def extract_text(self) -> str:
            return "Regional revenue by market"

    class FakeReader:
        def __init__(self, source) -> None:
            del source
            self.pages = [FakePage()]

    monkeypatch.setattr(parsers, "PDFTableExtractor", FakePDFTableExtractor)
    monkeypatch.setattr("pypdf.PdfReader", FakeReader)
    result = parsers.parse_document(
        "annual-report.pdf",
        _blank_pdf(),
        company_name="Example plc",
    )
    assert result.facts == []
    assert result.warnings == [
        "PDF_TABLES_FOUND_BUT_NO_SUPPORTED_STATEMENT_FACTS",
        "NO_NORMALIZED_FINANCIAL_FACTS_FOUND",
    ]

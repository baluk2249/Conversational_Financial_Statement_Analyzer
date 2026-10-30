from __future__ import annotations

from io import BytesIO

import pdfplumber
import pytest
from pypdf import PdfWriter

from application.service import AnalyzerService
from ingestion import (
    ExtractedTable,
    PDFFinancialFactAdapter,
    PDFTableExtractor,
    detect_statement_type,
)
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
            "CONSOLIDATED STATEMENT OF PROFIT AND LOSS",
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
            "CONSOLIDATED CASh FLO w STATEMENT",
        ),
    }
    for expected, headings in variants.items():
        assert all(detect_statement_type(heading) == expected for heading in headings)
    assert detect_statement_type(
        "CONSOLIDATED STATEMENT OF CASH FLOW; "
        "Net Profit Before Tax as per Statement of Profit and Loss"
    ) == "cash_flow_statement"
    assert detect_statement_type("Quarterly customer satisfaction scores 2025") == "other"


def test_consolidated_cash_flow_heading_is_included_in_scope() -> None:
    assert parsers._is_consolidated_statement_page(
        "CONSOLIDATED STATEMENT OF CASH FLOW"
    )


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


def test_pdf_table_extractor_prefers_populated_word_alignment_over_empty_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        width = 612
        height = 792

        def extract_text(self) -> str:
            return "CONSOLIDATED BALANCE SHEET"

        def extract_tables(self, table_settings=None):
            del table_settings
            return [[
                ["Particulars", "Note", "March 31, 2025", "March 31, 2024"],
            ]]

        def extract_words(self):
            rows = [
                [("Particulars", 50, 105), ("Note", 150, 170),
                 ("March", 210, 245), ("31,", 247, 260), ("2025", 265, 290),
                 ("March", 320, 355), ("31,", 357, 370), ("2024", 375, 400)],
                [("Total", 50, 75), ("assets", 78, 105), ("1,200", 265, 300),
                 ("1,000", 375, 410)],
                [("Total", 50, 75), ("equity", 78, 105), ("800", 265, 285),
                 ("700", 375, 395)],
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
    assert tables[0].extraction_metadata["extraction_method"] == "word_alignment"
    assert ["Total assets", "1,200", "1,000"] in tables[0].rows


def test_date_header_columns_and_indian_units_are_normalized() -> None:
    table = ExtractedTable(
        source_file="synthetic-indian-report.pdf",
        page_number=25,
        table_index=1,
        headers=[
            "Particulars",
            "Note",
            "For the Year ended",
            "For the Year ended",
        ],
        rows=[
            ["March 31, 2025", "March 31, 2024"],
            ["Total income", "1", "1,200.00", "1,000.00"],
            ["Finance costs", "2", "300.00", "250.00"],
            ["C. Profit/(Loss) for the year (A + B)", "", "900.00", "750.00"],
        ],
        raw_text="Particulars Note For the Year ended March 31, 2025 March 31, 2024",
        extraction_metadata={
            "page_text": (
                "CONSOLIDATED STATEMENT OF PROFIT AND LOSS\n"
                "for the Year ended March 31, 2025\n"
                "(Currency: Indian rupees in lakhs)"
            )
        },
    )

    facts = parsers._normalize_pdf_table(
        table,
        statement_type="income_statement",
        company_name="Synthetic Example",
        company_id="synthetic-example",
        source_id="source-test",
        default_period=None,
        page_text=table.extraction_metadata["page_text"],
    )

    assert {(fact.canonical_line_item, fact.fiscal_year, fact.value) for fact in facts} == {
        ("revenue", "FY2025", 1200),
        ("revenue", "FY2024", 1000),
        ("interest_expense", "FY2025", 300),
        ("interest_expense", "FY2024", 250),
        ("net_income", "FY2025", 900),
        ("net_income", "FY2024", 750),
    }
    assert all(fact.currency == "INR" for fact in facts)
    assert all(fact.unit == "lakhs" for fact in facts)


def test_split_date_headers_map_rows_with_and_without_note_numbers() -> None:
    table = ExtractedTable(
        source_file="synthetic-indian-report.pdf",
        page_number=125,
        table_index=1,
        headers=["Particulars", "Note", "For the Year ended For the Year ended"],
        rows=[
            ["March 31, 2025", "March 31, 2024"],
            ["Total income", "37,569.16", "31,813.11"],
            ["Finance costs", "31", "14,484.08", "14,220.59"],
        ],
        raw_text="Particulars Note For the Year ended March 31, 2025 March 31, 2024",
        extraction_metadata={
            "page_text": (
                "CONSOLIDATED STATEMENT OF PROFIT AND LOSS\n"
                "(Currency: Indian rupees in lakhs)"
            )
        },
    )

    facts = parsers._normalize_pdf_table(
        table,
        statement_type="income_statement",
        company_name="Synthetic Example",
        company_id="synthetic-example",
        source_id="source-test",
        default_period=None,
        page_text=table.extraction_metadata["page_text"],
    )

    assert {
        (fact.canonical_line_item, fact.fiscal_year, str(fact.value))
        for fact in facts
    } == {
        ("revenue", "FY2025", "37569.16"),
        ("revenue", "FY2024", "31813.11"),
        ("interest_expense", "FY2025", "14484.08"),
        ("interest_expense", "FY2024", "14220.59"),
    }


def test_single_supported_cash_flow_row_is_normalized() -> None:
    table = ExtractedTable(
        source_file="synthetic-indian-report.pdf",
        page_number=127,
        table_index=1,
        headers=["Particulars", "March 31, 2025", "March 31, 2024"],
        rows=[
            [
                "Net Cash (Used in) / Generated From Operating Activities",
                "12,799.22",
                "(2,615.64)",
            ]
        ],
        raw_text="Particulars March 31, 2025 March 31, 2024",
        extraction_metadata={
            "page_text": (
                "CONSOLIDATED CASH FLOW STATEMENT\n"
                "(Currency: Indian rupees in lakhs)"
            )
        },
    )

    facts = parsers._normalize_pdf_table(
        table,
        statement_type="cash_flow_statement",
        company_name="Synthetic Example",
        company_id="synthetic-example",
        source_id="source-test",
        default_period=None,
        page_text=table.extraction_metadata["page_text"],
    )

    assert {
        (fact.canonical_line_item, fact.fiscal_year, str(fact.value))
        for fact in facts
    } == {
        ("operating_cash_flow", "FY2025", "12799.22"),
        ("operating_cash_flow", "FY2024", "-2615.64"),
    }


def test_pdf_financial_adapter_handles_notes_and_side_by_side_tables() -> None:
    adapter = PDFFinancialFactAdapter()
    income_table = ExtractedTable(
        source_file="reliance-synthetic.pdf",
        page_number=55,
        table_index=1,
        headers=["CONSOLIDATED STATEMENT OF PROFIT AND LOSS", "Consolidated"],
        rows=[
            ["(INR in crore)", "(INR in crore)"],
            ["Notes", "2022-23", "2021-22", "Notes", "2022-23", "2021-22"],
            ["Income", "Net Profit Attributable to:"],
            [
                "Revenue from Operations", "25", "8,91,311", "7,17,635",
                "b) Non-Controlling Interest", "(29)", "75",
            ],
            [
                "Total Income", "9,03,045", "7,32,578",
                "a) Owners of the Company", "47,919", "82,890",
            ],
            [
                "Finance Costs", "29", "19,571", "14,584",
                "Diluted (in INR) - Before Exceptional Items", "33", "97.97", "84.12",
            ],
            ["Profit Before Tax *", "94,046", "82,154"],
            ["Profit for the Year", "74,088", "67,845"],
        ],
        raw_text="Notes 2022-23 2021-22",
        extraction_metadata={
            "page_text": (
                "CONSOLIDATED STATEMENT OF PROFIT AND LOSS\n"
                "For the year ended 31st March, 2023\n"
                "(INR in crore)"
            )
        },
    )
    income_facts = adapter.adapt_table(
        income_table,
        statement_type="income_statement",
        company_name="Synthetic Reliance",
        company_id="synthetic-reliance",
        source_id="source-income",
        default_period=None,
        page_text=income_table.extraction_metadata["page_text"],
    )

    revenue_facts = [
        fact for fact in income_facts
        if fact.canonical_line_item == "revenue"
    ]
    assert [(fact.fiscal_year, str(fact.value)) for fact in revenue_facts] == [
        ("FY2023", "891311"),
        ("FY2022", "717635"),
    ]
    assert all(fact.currency == "INR" for fact in revenue_facts)
    assert all(fact.unit == "crore" for fact in revenue_facts)
    assert revenue_facts[0].period_start.isoformat() == "2022-04-01"
    assert revenue_facts[0].period_end.isoformat() == "2023-03-31"

    cash_flow_table = ExtractedTable(
        source_file="reliance-synthetic.pdf",
        page_number=57,
        table_index=1,
        headers=["CONSOLIDATED STATEMENT OF CASH FLOW", "Consolidated"],
        rows=[
            ["(INR in crore)", "(INR in crore)"],
            ["2022-23", "2021-22", "2022-23", "2021-22"],
            ["A. Cash Flow from Operating Activities", "C. Cash Flow from Financing Activities"],
            ["Net Cash Flow from Operating Activities *", "1,15,032", "1,10,654"],
            ["Trade and Other Receivables", "13,194", "(14,180)"],
            ["Inventories", "(32,228)", "(24,983)"],
        ],
        raw_text="2022-23 2021-22",
        extraction_metadata={
            "page_text": (
                "CONSOLIDATED STATEMENT OF CASH FLOW\n"
                "For the year ended 31st March, 2023\n"
                "(INR in crore)"
            )
        },
    )
    cash_flow_facts = adapter.adapt_table(
        cash_flow_table,
        statement_type="cash_flow_statement",
        company_name="Synthetic Reliance",
        company_id="synthetic-reliance",
        source_id="source-cash-flow",
        default_period=None,
        page_text=cash_flow_table.extraction_metadata["page_text"],
    )
    assert [
        (fact.canonical_line_item, fact.fiscal_year, str(fact.value))
        for fact in cash_flow_facts
    ] == [
        ("operating_cash_flow", "FY2023", "115032"),
        ("operating_cash_flow", "FY2022", "110654"),
    ]


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

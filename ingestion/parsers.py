"""Parse supported uploads without guessing financial values or provenance."""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents.retrieval_grounding.config import METRIC_ALIASES
from models.schemas import DocumentChunk, FinancialFact, SourceLocator
from ingestion.pdf_tables import (
    ExtractedTable,
    PDFTableExtractionError,
    PDFTableExtractor,
    detect_statement_type,
)

logger = logging.getLogger(__name__)

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
_STATEMENT_ALIASES = {
    "income statement": "income_statement",
    "profit and loss": "income_statement",
    "p and l": "income_statement",
    "balance sheet": "balance_sheet",
    "cash flow": "cash_flow_statement",
    "cash flow statement": "cash_flow_statement",
}
_INCOME_ITEMS = {
    "revenue", "gross_profit", "operating_profit", "net_income",
    "ebitda", "ebit", "interest_expense", "cost_of_goods_sold",
}
_CASH_FLOW_ITEMS = {"operating_cash_flow"}


class DocumentProcessingError(ValueError):
    """Raised when a supported file cannot be safely normalized."""


class UnsupportedDocumentType(DocumentProcessingError):
    """Raised when an upload uses an unsupported file extension."""


@dataclass
class IngestionResult:
    source_id: str
    source_file: str
    facts: list[FinancialFact] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def periods(self) -> list[str]:
        return sorted({fact.fiscal_year for fact in self.facts} |
                      {chunk.period for chunk in self.chunks if chunk.period})

    @property
    def statement_types(self) -> list[str]:
        return sorted({fact.statement_type for fact in self.facts})


def parse_document(
    filename: str,
    content: bytes,
    *,
    company_name: str | None = None,
    company_id: str | None = None,
    period: str | None = None,
) -> IngestionResult:
    """Parse CSV/XLSX normalized rows or preserve PDF pages as narrative chunks."""
    safe_filename = Path(filename).name
    if not safe_filename or safe_filename in {".", ".."}:
        raise DocumentProcessingError("A valid file name is required.")
    if not content:
        raise DocumentProcessingError("The uploaded document is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DocumentProcessingError("The uploaded document exceeds the 20 MB limit.")

    extension = Path(safe_filename).suffix.casefold()
    if extension not in {".csv", ".xlsx", ".pdf"}:
        raise UnsupportedDocumentType(
            "Supported document types are PDF, XLSX, and CSV."
        )

    source_id = f"source-{uuid4().hex}"
    result = IngestionResult(source_id=source_id, source_file=safe_filename)
    if extension == ".pdf":
        chunks, facts, warnings = _parse_pdf(
            content,
            safe_filename,
            source_id,
            company_name=company_name,
            company_id=company_id,
            period=period,
        )
        result.chunks.extend(chunks)
        result.facts.extend(facts)
        result.warnings.extend(warnings)
        if not result.chunks:
            result.warnings.append("NO_EXTRACTABLE_NARRATIVE_TEXT")
        if not result.facts:
            result.warnings.append("NO_NORMALIZED_FINANCIAL_FACTS_FOUND")
        return result

    sheets = (
        [("CSV", _read_csv(content))]
        if extension == ".csv"
        else _read_xlsx(content)
    )
    for sheet_name, rows in sheets:
        result.facts.extend(
            _normalize_rows(
                rows,
                sheet_name=sheet_name,
                filename=safe_filename,
                source_id=source_id,
                company_name=company_name,
                company_id=company_id,
                default_period=period,
            )
        )
        result.chunks.extend(
            _narrative_chunks(
                rows,
                sheet_name=sheet_name,
                filename=safe_filename,
                source_id=source_id,
                company_name=company_name,
                company_id=company_id,
                default_period=period,
            )
        )
    if not result.facts and not result.chunks:
        raise DocumentProcessingError(
            "No supported financial rows or narrative text were found. "
            "Use columns for company, fiscal_year, statement_type, line_item, "
            "and value."
        )
    if not result.facts:
        result.warnings.append("NO_STRUCTURED_FINANCIAL_FACTS_FOUND")
    return result


def _read_csv(content: bytes) -> list[tuple[int, dict[str, Any]]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentProcessingError("CSV files must use UTF-8 encoding.") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise DocumentProcessingError("The CSV file must contain a header row.")
    return [
        (row_number, row)
        for row_number, row in enumerate(reader, start=2)
        if any(value is not None and str(value).strip() for value in row.values())
    ]


def _read_xlsx(content: bytes) -> list[tuple[str, list[tuple[int, dict[str, Any]]]]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise DocumentProcessingError(
            "XLSX processing requires the openpyxl dependency."
        ) from exc
    try:
        workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:
        raise DocumentProcessingError("The XLSX file could not be read.") from exc

    sheets: list[tuple[str, list[tuple[int, dict[str, Any]]]]] = []
    try:
        for worksheet in workbook.worksheets:
            values = worksheet.iter_rows(values_only=True)
            headers = next(values, None)
            if not headers:
                continue
            names = [str(value).strip() if value is not None else "" for value in headers]
            rows = []
            for row_number, values_row in enumerate(values, start=2):
                if not any(value is not None and str(value).strip() for value in values_row):
                    continue
                rows.append((row_number, dict(zip(names, values_row, strict=False))))
            sheets.append((worksheet.title, rows))
    finally:
        workbook.close()
    return sheets


def _normalize_rows(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    sheet_name: str,
    filename: str,
    source_id: str,
    company_name: str | None,
    company_id: str | None,
    default_period: str | None,
    statement_type_override: str | None = None,
    page_number: int | None = None,
    table_index: int | None = None,
) -> list[FinancialFact]:
    facts: list[FinancialFact] = []
    for row_number, raw_row in rows:
        row = {_header_key(key): value for key, value in raw_row.items()}
        label = _first(row, "line_item", "metric", "account", "label")
        value = _first(row, "value", "amount", "reported_value")
        canonical = _canonical_metric(label)
        if canonical is None or value is None or not str(value).strip():
            continue

        row_company_name = _nonempty(_first(row, "company_name", "company")) or company_name
        row_company_id = _nonempty(_first(row, "company_id")) or company_id
        if not row_company_name and row_company_id:
            row_company_name = row_company_id
        if not row_company_id and row_company_name:
            row_company_id = _slug(row_company_name)
        fiscal_year = _nonempty(
            _first(row, "fiscal_year", "period", "reporting_period")
        ) or default_period
        if not row_company_name or not row_company_id or not fiscal_year:
            raise DocumentProcessingError(
                f"Row {row_number} in {sheet_name} needs company identity and fiscal_year."
            )
        statement_type = statement_type_override or _statement_type(
            _first(row, "statement_type", "statement")
        ) or _default_statement(canonical)
        try:
            amount = _decimal_value(value)
        except (InvalidOperation, ValueError) as exc:
            raise DocumentProcessingError(
                f"Row {row_number} in {sheet_name} has an invalid financial value."
            ) from exc

        start = _parse_date(_first(row, "period_start", "start_date"))
        end = _parse_date(_first(row, "period_end", "end_date"))
        fiscal_year_end = _parse_date(_first(row, "fiscal_year_end"))
        currency = _nonempty(_first(row, "currency"))
        unit = _nonempty(_first(row, "unit")) or "absolute"
        cell = _row_locator(raw_row, row_number)
        if page_number is not None:
            locator = SourceLocator(
                page=page_number,
                label=str(label).strip(),
                details={
                    "table_index": table_index,
                    "table_row": row_number,
                },
            )
        else:
            locator = SourceLocator(
                label=str(label).strip(),
                sheet=sheet_name,
                cell_range=cell,
            )
        facts.append(
            FinancialFact(
                fact_id=f"fact-{uuid4().hex}",
                company_id=row_company_id,
                company_name=row_company_name,
                statement_type=statement_type,
                canonical_line_item=canonical,
                metric_definition=_nonempty(_first(row, "metric_definition")),
                period_start=start,
                period_end=end,
                fiscal_year=fiscal_year,
                fiscal_year_end=fiscal_year_end,
                value=amount,
                currency=currency.upper() if currency else None,
                unit=unit,
                source_id=source_id,
                source_file=filename,
                locator=locator,
                extraction_confidence=1.0,
            )
        )
    return facts


def _narrative_chunks(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    sheet_name: str,
    filename: str,
    source_id: str,
    company_name: str | None,
    company_id: str | None,
    default_period: str | None,
) -> list[DocumentChunk]:
    chunks = []
    for row_number, raw_row in rows:
        row = {_header_key(key): value for key, value in raw_row.items()}
        narrative = _nonempty(_first(row, "narrative", "commentary", "notes", "text"))
        if not narrative:
            continue
        row_company_name = _nonempty(_first(row, "company_name", "company")) or company_name
        row_company_id = _nonempty(_first(row, "company_id")) or company_id
        if not row_company_name and row_company_id:
            row_company_name = row_company_id
        if not row_company_id and row_company_name:
            row_company_id = _slug(row_company_name)
        if not row_company_name or not row_company_id:
            raise DocumentProcessingError(
                f"Narrative row {row_number} in {sheet_name} needs a company identity."
            )
        chunks.append(
            DocumentChunk(
                chunk_id=f"chunk-{uuid4().hex}",
                company_id=row_company_id,
                company_name=row_company_name,
                document_type=_nonempty(_first(row, "document_type")) or "financial_statement",
                statement_type=_statement_type(_first(row, "statement_type", "statement")),
                section=_nonempty(_first(row, "section")) or "Narrative Notes",
                period=_nonempty(
                    _first(row, "fiscal_year", "period", "reporting_period")
                ) or default_period,
                source_id=source_id,
                source_file=filename,
                sheet=sheet_name,
                cell_range=_row_locator(raw_row, row_number),
                text=narrative,
            )
        )
    return chunks


def _parse_pdf(
    content: bytes,
    filename: str,
    source_id: str,
    *,
    company_name: str | None,
    company_id: str | None,
    period: str | None,
) -> tuple[list[DocumentChunk], list[FinancialFact], list[str]]:
    if not company_name and not company_id:
        raise DocumentProcessingError(
            "Provide a company name or ID to scope PDF narrative evidence."
        )
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentProcessingError("PDF processing requires the pypdf dependency.") from exc
    try:
        reader = PdfReader(io.BytesIO(content))
        page_text = [(index + 1, (page.extract_text() or "").strip())
                     for index, page in enumerate(reader.pages)]
    except Exception as exc:
        raise DocumentProcessingError("The PDF file could not be read.") from exc

    try:
        extracted_tables = PDFTableExtractor().extract_tables(
            io.BytesIO(content), source_file=filename
        )
    except PDFTableExtractionError as exc:
        raise DocumentProcessingError(str(exc)) from exc

    scoped_company_id = company_id or _slug(company_name or "")
    scoped_company_name = company_name or scoped_company_id
    chunks = []
    for page_number, text in page_text:
        if not text:
            continue
        page_tables = [
            table for table in extracted_tables
            if table.page_number == page_number
        ]
        page_statement_types = {
            detect_statement_type(
                f"{table.extraction_metadata.get('page_text', '')}\n{table.raw_text}"
            )
            for table in page_tables
        }
        page_statement_type = (
            next(iter(page_statement_types))
            if len(page_statement_types) == 1
            and "other" not in page_statement_types
            else None
        )
        chunks.append(
            DocumentChunk(
                chunk_id=f"chunk-{uuid4().hex}",
                company_id=scoped_company_id,
                company_name=scoped_company_name,
                document_type="annual_report",
                section=page_statement_type or "PDF page",
                period=period or _single_fiscal_year(text),
                statement_type=page_statement_type,
                source_id=source_id,
                source_file=filename,
                page=page_number,
                text=text,
            )
        )

    facts: list[FinancialFact] = []
    warnings: list[str] = []
    recognized_tables = 0
    for table in extracted_tables:
        page_source_text = str(
            table.extraction_metadata.get("page_text", "")
        )
        statement_type = detect_statement_type(
            f"{page_source_text}\n{table.raw_text}"
        )
        if statement_type == "other":
            continue
        table_facts = _normalize_pdf_table(
            table,
            statement_type=statement_type,
            company_name=scoped_company_name,
            company_id=scoped_company_id,
            source_id=source_id,
            default_period=period,
            page_text=page_source_text,
        )
        if not table_facts:
            continue
        facts.extend(table_facts)
        recognized_tables += 1

    if extracted_tables and not recognized_tables:
        warnings.append("PDF_TABLES_FOUND_BUT_NO_SUPPORTED_STATEMENT_FACTS")
    if not extracted_tables:
        warnings.append("NO_PDF_TABLES_EXTRACTED")
    return chunks, facts, warnings


def _normalize_pdf_table(
    table: ExtractedTable,
    *,
    statement_type: str,
    company_name: str,
    company_id: str,
    source_id: str,
    default_period: str | None,
    page_text: str,
) -> list[FinancialFact]:
    supported_rows = {
        _canonical_metric(row[0])
        for row in table.rows
        if row and row[0] and _canonical_metric(row[0]) is not None
    }
    if len(supported_rows) < 2:
        return []

    period_columns = _period_columns(
        table, page_text=page_text, default_period=default_period
    )
    if not period_columns:
        return []

    context = "\n".join(
        [table.raw_text, page_text, *table.headers]
    )
    currency = _currency_code(context)
    unit = _financial_unit(context)
    table_rows: list[tuple[int, dict[str, Any]]] = []
    for row_number, raw_cells in enumerate(table.rows, start=2):
        if not raw_cells:
            continue
        label = raw_cells[0]
        if not label or _canonical_metric(label) is None:
            continue
        for column, fiscal_year in period_columns.items():
            if column >= len(raw_cells):
                continue
            value = _pdf_amount(raw_cells[column])
            if value is None:
                continue
            table_rows.append(
                (
                    row_number,
                    {
                        "company_id": company_id,
                        "company_name": company_name,
                        "fiscal_year": fiscal_year,
                        "statement_type": statement_type,
                        "line_item": label,
                        "value": str(value),
                        "currency": currency,
                        "unit": unit,
                    },
                )
            )
    if not table_rows:
        return []

    return _normalize_rows(
        table_rows,
        sheet_name="",
        filename=table.source_file,
        source_id=source_id,
        company_name=company_name,
        company_id=company_id,
        default_period=default_period,
        statement_type_override=statement_type,
        page_number=table.page_number,
        table_index=table.table_index,
    )


def _period_columns(
    table: ExtractedTable,
    *,
    page_text: str,
    default_period: str | None,
) -> dict[int, str]:
    candidates = [table.headers, *table.rows[:5]]
    best: dict[int, str] = {}
    for row in candidates:
        years = {
            index: period
            for index, cell in enumerate(row)
            if cell and (period := _fiscal_period(cell)) is not None
        }
        if len(years) > len(best):
            best = years

    numeric_columns = _numeric_table_columns(table)
    if best:
        return {column: year for column, year in best.items() if column in numeric_columns}

    inferred_period = default_period or _single_fiscal_year(page_text)
    if inferred_period and len(numeric_columns) == 1:
        return {numeric_columns[0]: inferred_period}
    return {}


def _numeric_table_columns(table: ExtractedTable) -> set[int]:
    numeric_columns: set[int] = set()
    for row in table.rows:
        if not row or _canonical_metric(row[0]) is None:
            continue
        for index, cell in enumerate(row[1:], start=1):
            if _pdf_amount(cell) is not None:
                numeric_columns.add(index)
    return numeric_columns


def _fiscal_period(value: str) -> str | None:
    match = re.search(
        r"\b(?:FY\s*)?(20\d{2})(?:\s*[/\-]\s*(?:\d{2}|20\d{2}))?\b",
        value,
        re.IGNORECASE,
    )
    if not match:
        return None
    year = match.group(1)
    ending_year = re.search(r"[/\-]\s*(\d{2}|20\d{2})\b", match.group(0))
    if ending_year:
        suffix = ending_year.group(1)
        year = suffix if len(suffix) == 4 else f"20{suffix}"
    return f"FY{year}"


def _pdf_amount(value: Any) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.casefold() in {"", "-", "–", "—", "−", "n/a", "na", "nil"}:
        return None
    if re.fullmatch(r"[\d,().+\-£$€¥\s]+", text) is None:
        return None
    try:
        return _decimal_value(text)
    except (InvalidOperation, ValueError):
        return None


def _currency_code(text: str) -> str | None:
    if "£" in text or re.search(r"\bGBP\b", text, re.IGNORECASE):
        return "GBP"
    if "$" in text or re.search(r"\bUSD\b", text, re.IGNORECASE):
        return "USD"
    if "€" in text or re.search(r"\bEUR\b", text, re.IGNORECASE):
        return "EUR"
    if "¥" in text or re.search(r"\bJPY\b", text, re.IGNORECASE):
        return "JPY"
    return None


def _financial_unit(text: str) -> str:
    normalized = text.casefold()
    if re.search(r"\b(?:in\s+)?(?:millions?|mn)\b|[£$€¥]\s?m\b", normalized):
        return "millions"
    if re.search(r"\b(?:in\s+)?(?:thousands?|000s?)\b|[£$€¥]\s?000\b", normalized):
        return "thousands"
    return "absolute"


def _canonical_metric(value: Any) -> str | None:
    if value is None:
        return None
    label = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    for canonical, aliases in METRIC_ALIASES.items():
        candidates = (canonical.replace("_", " "), *aliases)
        if any(label == re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
               for alias in candidates):
            return canonical
    return None


def _statement_type(value: Any) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    return _STATEMENT_ALIASES.get(normalized, normalized.replace(" ", "_"))


def _default_statement(metric: str) -> str:
    if metric in _INCOME_ITEMS:
        return "income_statement"
    if metric in _CASH_FLOW_ITEMS:
        return "cash_flow_statement"
    return "balance_sheet"


def _header_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _nonempty(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _decimal_value(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    text = str(value).strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(",", "")
    text = re.sub(r"^[£$€¥]\s*", "", text)
    text = re.sub(r"\s*[£$€¥]$", "", text)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        raise InvalidOperation
    amount = Decimal(text)
    return -amount if negative else amount


def _parse_date(value: Any) -> date | None:
    if value is None or not str(value).strip():
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise DocumentProcessingError("Dates must use YYYY-MM-DD format.") from exc


def _row_locator(row: dict[str, Any], row_number: int) -> str:
    headers = {_header_key(key): index + 1 for index, key in enumerate(row)}
    line_column = next(
        (headers[name] for name in ("line_item", "metric", "account", "label")
         if name in headers),
        1,
    )
    value_column = next(
        (headers[name] for name in ("value", "amount", "reported_value")
         if name in headers),
        line_column,
    )
    return f"{_column_letter(line_column)}{row_number}:{_column_letter(value_column)}{row_number}"


def _column_letter(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _single_fiscal_year(text: str) -> str | None:
    periods = {
        re.sub(r"\s+", "", match).upper()
        for match in re.findall(r"\bFY\s?\d{4}\b", text, flags=re.IGNORECASE)
    }
    return next(iter(periods)) if len(periods) == 1 else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")

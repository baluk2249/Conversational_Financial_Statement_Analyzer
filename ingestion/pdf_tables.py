"""Extract page-scoped table grids from PDFs without normalizing their meaning."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedTable:
    source_file: str
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str | None]]
    raw_text: str
    extraction_metadata: dict[str, Any] = field(default_factory=dict)


class PDFTableExtractionError(RuntimeError):
    """Raised when a PDF table extraction operation fails."""


class PDFTableExtractor:
    """Extract tables using pdfplumber while retaining page/table provenance."""

    def extract_tables(
        self,
        file_path: str | Path | BinaryIO,
        *,
        source_file: str | None = None,
    ) -> list[ExtractedTable]:
        try:
            import pdfplumber
        except ImportError as exc:
            raise PDFTableExtractionError(
                "PDF table extraction requires the pdfplumber dependency."
            ) from exc

        filename = source_file or _source_name(file_path)
        try:
            with pdfplumber.open(file_path) as document:
                extracted: list[ExtractedTable] = []
                for page_number, page in enumerate(document.pages, start=1):
                    page_text = page.extract_text() or ""
                    raw_tables = page.extract_tables() or []
                    extraction_method = "pdfplumber"
                    if not raw_tables:
                        raw_tables = page.extract_tables(
                            table_settings={
                                "vertical_strategy": "text",
                                "horizontal_strategy": "text",
                                "min_words_vertical": 1,
                                "min_words_horizontal": 1,
                            }
                        ) or []
                        extraction_method = "pdfplumber_text"
                    if not _has_period_header(raw_tables):
                        word_table = _extract_word_aligned_table(page)
                        if word_table and _has_period_header([word_table]):
                            raw_tables = [word_table]
                            extraction_method = "word_alignment"
                    seen_tables: set[tuple[tuple[str | None, ...], ...]] = set()
                    table_index = 0
                    for raw_rows in raw_tables:
                        normalized_rows = [
                            [_clean_cell(cell) for cell in row]
                            for row in raw_rows
                            if row and any(_clean_cell(cell) for cell in row)
                        ]
                        if not normalized_rows:
                            continue
                        signature = tuple(tuple(row) for row in normalized_rows)
                        if signature in seen_tables:
                            continue
                        seen_tables.add(signature)
                        table_index += 1
                        headers = normalized_rows[0]
                        rows = normalized_rows[1:]
                        extracted.append(
                            ExtractedTable(
                                source_file=filename,
                                page_number=page_number,
                                table_index=table_index,
                                headers=headers,
                                rows=rows,
                                raw_text="\n".join(
                                    "\t".join(cell or "" for cell in row)
                                    for row in normalized_rows
                                ),
                                extraction_metadata={
                                    "page_text": page_text,
                                    "page_width": page.width,
                                    "page_height": page.height,
                                    "extraction_method": extraction_method,
                                },
                            )
                        )
                return extracted
        except PDFTableExtractionError:
            raise
        except Exception as exc:
            raise PDFTableExtractionError(
                f"Could not extract tables from {filename}"
            ) from exc


_STATEMENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "income_statement",
        re.compile(
            r"\b(?:income statement|statement of (?:profit and loss|operations|income)|"
            r"profit and loss account|profit & loss account|profit or loss)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "balance_sheet",
        re.compile(
            r"\b(?:balance sheet|statement of financial position)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cash_flow_statement",
        re.compile(
            r"\b(?:cash flow statement|statement of cash flows?|"
            r"consolidated statement of cash flows?)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_statement_type(text: str) -> str:
    """Classify only explicit supported statement headings; never numeric tables alone."""
    matches = [
        statement_type
        for statement_type, pattern in _STATEMENT_PATTERNS
        if pattern.search(text)
    ]
    return matches[0] if len(matches) == 1 else "other"


def _clean_cell(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _has_period_header(tables: list[list[list[Any]]]) -> bool:
    for rows in tables:
        for row in rows[:5]:
            years = {
                match.group(0)
                for cell in row
                if cell is not None
                for match in re.finditer(r"\b20\d{2}\b", str(cell))
            }
            if len(years) > 1 or (
                years
                and any(
                    re.search(r"\b(?:fy|year ended|as of|as at)\b", str(cell), re.I)
                    for cell in row
                    if cell is not None
                )
            ):
                return True
    return False


def _extract_word_aligned_table(page: Any) -> list[list[str]] | None:
    words = page.extract_words()
    if not words:
        return None

    lines: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        if not lines or abs(top - float(lines[-1][0]["top"])) > 3:
            lines.append([word])
        else:
            lines[-1].append(word)

    rows: list[list[str]] = []
    for line in lines:
        cells: list[list[str]] = []
        right_edge: float | None = None
        for word in sorted(line, key=lambda item: float(item["x0"])):
            left = float(word["x0"])
            if right_edge is None or left - right_edge >= 16:
                cells.append([str(word["text"])])
            else:
                cells[-1].append(str(word["text"]))
            right_edge = float(word["x1"])
        if len(cells) >= 2:
            rows.append([" ".join(cell) for cell in cells])

    return rows if len(rows) >= 2 else None


def _source_name(file_path: str | Path | BinaryIO) -> str:
    if isinstance(file_path, (str, Path)):
        return Path(file_path).name
    return Path(getattr(file_path, "name", "uploaded.pdf")).name

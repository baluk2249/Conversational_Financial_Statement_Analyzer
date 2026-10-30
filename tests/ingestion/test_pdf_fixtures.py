from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pdfplumber
from numpy.typing import NDArray

from agents.trend_red_flag.red_flags import evaluate_red_flags
from agents.trend_red_flag.rules import load_rules
from agents.trend_red_flag.trends import analyse_trends
from finance.ratios import calculate_ratio
from ingestion import PDFTableExtractor, parse_document
from retrieval.faiss_store import FaissNarrativeStore


FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "pdfs"
EXPECTED_FILENAMES = {
    "sample_complete_financials.pdf",
    "sample_income_statement.pdf",
    "sample_balance_sheet.pdf",
    "sample_cash_flow.pdf",
    "sample_missing_values.pdf",
    "sample_units_millions.pdf",
    "sample_ambiguous_table.pdf",
    "sample_company_northstar_financials.pdf",
    "sample_company_cedar_financials.pdf",
    "sample_company_orbit_financials.pdf",
}


class FixtureEmbedder:
    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        terms = ("revenue", "expenses", "inventory", "receivables", "cash flow")
        return np.asarray(
            [[float(text.casefold().count(term)) for term in terms] for text in texts],
            dtype=np.float32,
        )


def test_fixture_pdfs_exist_and_contain_selectable_text() -> None:
    assert {path.name for path in FIXTURE_DIRECTORY.glob("*.pdf")} == EXPECTED_FILENAMES

    for filename in EXPECTED_FILENAMES:
        with pdfplumber.open(FIXTURE_DIRECTORY / filename) as pdf:
            assert pdf.pages
            assert all(page.extract_text() for page in pdf.pages)


def test_complete_fixture_extracts_tables_normalizes_facts_and_indexes_commentary() -> None:
    fixture_path = FIXTURE_DIRECTORY / "sample_complete_financials.pdf"
    tables = PDFTableExtractor().extract_tables(fixture_path)
    financial_tables = [
        table for table in tables
        if {"FY2024", "FY2025"}.issubset(set(table.headers))
    ]
    assert len(financial_tables) == 3
    assert [table.page_number for table in financial_tables] == [1, 2, 3]

    with pdfplumber.open(fixture_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    for heading in (
        "CONSOLIDATED INCOME STATEMENT",
        "CONSOLIDATED BALANCE SHEET",
        "CONSOLIDATED CASH FLOW STATEMENT",
        "MANAGEMENT COMMENTARY",
    ):
        assert heading in full_text

    result = parse_document(
        fixture_path.name,
        fixture_path.read_bytes(),
        company_name="ABC DEMO LTD",
    )
    assert result.periods == ["FY2024", "FY2025"]
    assert result.statement_types == [
        "balance_sheet",
        "cash_flow_statement",
        "income_statement",
    ]
    assert result.facts
    assert {fact.fiscal_year for fact in result.facts} == {"FY2024", "FY2025"}
    assert {
        (fact.statement_type, fact.locator.page) for fact in result.facts
    } == {
        ("income_statement", 1),
        ("balance_sheet", 2),
        ("cash_flow_statement", 3),
    }

    revenue_2025 = next(
        fact for fact in result.facts
        if fact.canonical_line_item == "revenue" and fact.fiscal_year == "FY2025"
    )
    assert str(revenue_2025.value) == "120.0"
    assert revenue_2025.currency == "GBP"
    assert revenue_2025.unit == "millions"
    assert revenue_2025.locator.page == 1

    commentary = next(chunk for chunk in result.chunks if chunk.page == 4)
    assert "operating cash flow declined" in commentary.text.casefold()
    vector_store = FaissNarrativeStore(embedder=FixtureEmbedder())
    vector_store.add_chunks(result.chunks)
    search_results = vector_store.search(
        "revenue expenses inventory receivables cash flow",
        company_id="abc-demo-ltd",
        top_k=5,
    )
    assert any(chunk.page == 4 for chunk in search_results)


def test_missing_values_remain_blank_and_are_not_normalized_as_zero() -> None:
    fixture_path = FIXTURE_DIRECTORY / "sample_missing_values.pdf"
    tables = PDFTableExtractor().extract_tables(fixture_path)
    assert len(tables) == 1
    assert tables[0].rows[0][2] is None

    result = parse_document(
        fixture_path.name,
        fixture_path.read_bytes(),
        company_name="ABC DEMO LTD",
    )
    assert len(result.facts) == 6
    assert {
        (fact.canonical_line_item, fact.fiscal_year) for fact in result.facts
    } == {
        ("revenue", "FY2025"),
        ("cost_of_goods_sold", "FY2024"),
        ("gross_profit", "FY2025"),
        ("operating_profit", "FY2024"),
        ("interest_expense", "FY2025"),
        ("net_income", "FY2024"),
    }
    assert all(fact.value != 0 for fact in result.facts)


def test_statement_specific_fixtures_and_explicit_units_are_extractable() -> None:
    expected_headings = {
        "sample_income_statement.pdf": "CONSOLIDATED INCOME STATEMENT",
        "sample_balance_sheet.pdf": "CONSOLIDATED BALANCE SHEET",
        "sample_cash_flow.pdf": "CONSOLIDATED CASH FLOW STATEMENT",
        "sample_units_millions.pdf": "CONSOLIDATED INCOME STATEMENT",
    }
    for filename, heading in expected_headings.items():
        with pdfplumber.open(FIXTURE_DIRECTORY / filename) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        assert heading in text
        assert "GBP millions" in text

    units_path = FIXTURE_DIRECTORY / "sample_units_millions.pdf"
    units_result = parse_document(
        units_path.name,
        units_path.read_bytes(),
        company_name="ABC DEMO LTD",
    )
    assert units_result.facts
    assert all(fact.currency == "GBP" for fact in units_result.facts)
    assert all(fact.unit == "millions" for fact in units_result.facts)
    assert next(
        fact for fact in units_result.facts
        if fact.canonical_line_item == "revenue" and fact.fiscal_year == "FY2025"
    ).value == 120


def test_ambiguous_table_does_not_normalize_financial_facts() -> None:
    fixture_path = FIXTURE_DIRECTORY / "sample_ambiguous_table.pdf"
    with pdfplumber.open(fixture_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    assert "FY202" not in text
    assert "GBP" not in text
    assert "USD" not in text
    result = parse_document(
        fixture_path.name,
        fixture_path.read_bytes(),
        company_name="Synthetic Unclassified Data",
    )
    assert result.facts == []


def test_three_company_pdfs_are_selectable_and_company_scoped() -> None:
    expected_companies = {
        "sample_company_northstar_financials.pdf": "NORTHSTAR DEMO LTD",
        "sample_company_cedar_financials.pdf": "CEDAR DEMO PLC",
        "sample_company_orbit_financials.pdf": "ORBIT DEMO INC",
    }

    company_ids = set()
    for filename, company_name in expected_companies.items():
        fixture_path = FIXTURE_DIRECTORY / filename
        with pdfplumber.open(fixture_path) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
        assert len(pages_text) == 4
        assert all("SYNTHETIC TEST DATA ONLY" in text for text in pages_text)
        assert all(company_name in text for text in pages_text)
        assert "CONSOLIDATED INCOME STATEMENT" in pages_text[0]
        assert "CONSOLIDATED BALANCE SHEET" in pages_text[1]
        assert "CONSOLIDATED CASH FLOW STATEMENT" in pages_text[2]
        assert "MANAGEMENT COMMENTARY" in pages_text[3]

        result = parse_document(
            fixture_path.name,
            fixture_path.read_bytes(),
            company_name=company_name,
        )
        periods = ["FY2023", "FY2024", "FY2025"]
        assert {fact.fiscal_year for fact in result.facts} == set(periods)
        assert {fact.company_name for fact in result.facts} == {company_name}
        assert result.facts
        assert all(
            fact.currency == "GBP"
            and fact.unit == "millions"
            and fact.period_start == date(int(fact.fiscal_year[-4:]), 1, 1)
            and fact.period_end == date(int(fact.fiscal_year[-4:]), 12, 31)
            and fact.fiscal_year_end == date(int(fact.fiscal_year[-4:]), 12, 31)
            for fact in result.facts
        )
        assert {
            (fact.statement_type, fact.locator.page) for fact in result.facts
        } == {
            ("income_statement", 1),
            ("balance_sheet", 2),
            ("cash_flow_statement", 3),
        }
        metrics = [
            calculate_ratio(result.facts, result.facts[0].company_id, metric, period)
            for period in periods[1:]
            for metric in (
                "current_ratio",
                "quick_ratio",
                "operating_cash_flow_ratio",
                "debt_to_equity",
                "debt_to_assets",
                "interest_coverage",
                "gross_margin",
                "operating_margin",
                "net_margin",
                "roa",
                "roe",
                "dso",
                "inventory_days",
                "asset_turnover",
            )
        ]
        assert all(metric.quality_status == "VALID" for metric in metrics)
        assert all(
            metric.value is not None and metric.value > 0
            for metric in metrics
            if metric.metric_name in {"interest_coverage", "dso", "inventory_days"}
        )
        trends = [
            trend
            for metric_name in (
                "dso",
                "gross_margin",
                "operating_margin",
                "net_margin",
                "debt_to_equity",
                "debt_to_assets",
                "interest_coverage",
                "current_ratio",
                "quick_ratio",
                "operating_cash_flow_ratio",
            )
            for trend in analyse_trends(
                result.facts[0].company_id, metric_name, periods, metrics
            )
        ]
        red_flags = evaluate_red_flags(
            result.facts[0].company_id,
            periods,
            result.facts,
            metrics,
            trends,
            load_rules(),
        )
        assert len(red_flags) == 6
        assert all(flag.status != "INSUFFICIENT_EVIDENCE" for flag in red_flags)
        assert all(flag.supporting_refs for flag in red_flags)
        company_ids.add(result.facts[0].company_id)

    assert len(company_ids) == 3

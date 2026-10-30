from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
import numpy as np
from numpy.typing import NDArray

from agents.retrieval_grounding.agent import RetrievalGroundingAgent
from models.schemas import (
    DocumentChunk,
    FinancialFact,
    RedFlagResult,
    TrendResult,
)
from retrieval.faiss_store import FaissNarrativeStore
from retrieval.interfaces import NarrativeStoreUnavailableError
from retrieval.structured_store import SQLiteFinancialStore


def make_fact(
    company: str,
    metric: str,
    period: str,
    value: str,
    *,
    currency: str = "GBP",
    unit: str = "absolute",
    statement: str = "income_statement",
) -> FinancialFact:
    year = int(period[-4:])
    return FinancialFact(
        fact_id=f"{company}-{metric}-{period}",
        company_id=company,
        company_name=company.upper(),
        statement_type=statement,
        canonical_line_item=metric,
        period_start=date(year, 1, 1),
        period_end=date(year, 12, 31),
        fiscal_year=period,
        fiscal_year_end=date(year, 12, 31),
        value=Decimal(value),
        currency=currency,
        unit=unit,
        source_id=f"source-{company}-{period}",
        source_file=f"{company}-{period}.pdf",
        locator={"page": 12, "label": metric.replace("_", " ").title()},
        extraction_confidence=0.99,
    )


@pytest.fixture
def store() -> SQLiteFinancialStore:
    return SQLiteFinancialStore(":memory:")


def test_exact_revenue_retrieval_and_citation(store: SQLiteFinancialStore) -> None:
    store.add_fact(make_fact("abc", "revenue", "FY2025", "12000000.00"))
    result = RetrievalGroundingAgent(store).retrieve_context("What was revenue in FY2025?", "abc")
    assert result.status == "OK"
    assert result.facts[0].value == Decimal("12000000.00")
    assert result.facts[0].source_reference.citation == "[Income Statement | FY2025 | Revenue | p.12]"
    assert result.facts[0].currency == "GBP"


def test_company_and_period_isolation(store: SQLiteFinancialStore) -> None:
    store.add_fact(make_fact("abc", "revenue", "FY2024", "100"))
    store.add_fact(make_fact("abc", "revenue", "FY2025", "200"))
    store.add_fact(make_fact("xyz", "revenue", "FY2025", "999"))
    agent = RetrievalGroundingAgent(store)
    result = agent.retrieve_context("What was revenue in FY2025?", "abc")
    assert [(fact.company_id, fact.fiscal_year, fact.value) for fact in result.facts] == [
        ("abc", "FY2025", Decimal("200"))
    ]
    assert store.get_fact("xyz", "revenue", "FY2025").value == Decimal("999")


def test_statement_type_filtering(store: SQLiteFinancialStore) -> None:
    store.add_fact(make_fact("abc", "revenue", "FY2025", "100", statement="income_statement"))
    other_statement = make_fact("abc", "revenue", "FY2025", "999", statement="other_statement")
    store.add_fact(other_statement.model_copy(update={"fact_id": "abc-revenue-other-FY2025"}))
    result = store.get_fact("abc", "revenue", "FY2025", statement_type="income_statement")
    assert result.value == Decimal("100")


def test_current_ratio_is_calculated_deterministically_with_refs(store: SQLiteFinancialStore) -> None:
    store.add_fact(make_fact("abc", "current_assets", "FY2025", "150", statement="balance_sheet"))
    store.add_fact(make_fact("abc", "current_liabilities", "FY2025", "100", statement="balance_sheet"))
    result = RetrievalGroundingAgent(store).retrieve_context("What was the current ratio in FY2025?", "abc")
    calculated = next(metric for metric in result.computed_metrics if metric.metric_name == "current_ratio")
    assert result.status == "OK"
    assert calculated.value == Decimal("1.5000")
    assert calculated.formula_version == "current_ratio_v1"
    assert set(calculated.input_fact_refs) == {fact.fact_id for fact in result.facts}


def test_comparison_context_includes_prior_fact_provenance_without_extra_metrics(
    store: SQLiteFinancialStore,
) -> None:
    for period, assets, receivables, revenue, net_income in (
        ("FY2024", "80", "15", "100", "10"),
        ("FY2025", "120", "20", "120", "20"),
    ):
        store.add_fact(
            make_fact(
                "abc", "total_assets", period, assets, statement="balance_sheet"
            )
        )
        store.add_fact(
            make_fact(
                "abc",
                "accounts_receivable",
                period,
                receivables,
                statement="balance_sheet",
            )
        )
        store.add_fact(make_fact("abc", "revenue", period, revenue))
        store.add_fact(make_fact("abc", "net_income", period, net_income))

    result = RetrievalGroundingAgent(store).retrieve_context(
        "What are DSO and ROA in FY2025?",
        "abc",
        "FY2025",
        include_all_financial_facts=True,
    )

    assert {fact.fiscal_year for fact in result.facts} == {"FY2024", "FY2025"}
    assert {metric.period for metric in result.computed_metrics} == {"FY2025"}
    fact_ids = {fact.fact_id for fact in result.facts}
    assert all(
        set(metric.input_fact_refs) <= fact_ids
        for metric in result.computed_metrics
    )
    assert "FISCAL_YEAR_END_MISMATCH" not in result.warnings


def test_currency_and_unit_mismatch_warn_and_prevent_calculation(store: SQLiteFinancialStore) -> None:
    store.add_fact(make_fact("abc", "current_assets", "FY2025", "150", currency="GBP", unit="thousands"))
    store.add_fact(make_fact("abc", "current_liabilities", "FY2025", "100", currency="USD", unit="millions"))
    result = RetrievalGroundingAgent(store).retrieve_context("What was the current ratio in FY2025?", "abc")
    calculated = next(metric for metric in result.computed_metrics if metric.metric_name == "current_ratio")
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert "CURRENCY_MISMATCH" in result.warnings
    assert "UNIT_MISMATCH" in result.warnings
    assert calculated.value is None
    assert calculated.quality_status == "INCOMPATIBLE_INPUTS"


def test_fiscal_period_and_definition_mismatches_warn(store: SQLiteFinancialStore) -> None:
    assets = make_fact("abc", "current_assets", "FY2025", "150", statement="balance_sheet").model_copy(
        update={"period_start": date(2025, 1, 1), "period_end": date(2025, 12, 31),
                "fiscal_year_end": date(2025, 12, 31), "metric_definition": "consolidated"}
    )
    liabilities = make_fact("abc", "current_liabilities", "FY2025", "100", statement="balance_sheet").model_copy(
        update={"period_start": date(2025, 1, 1), "period_end": date(2025, 6, 30),
                "fiscal_year_end": date(2025, 6, 30), "metric_definition": "consolidated"}
    )
    conflicting_assets = assets.model_copy(
        update={"fact_id": "abc-current-assets-alternate", "metric_definition": "separate"}
    )
    store.add_fact(assets)
    store.add_fact(liabilities)
    store.add_fact(conflicting_assets)
    result = RetrievalGroundingAgent(store).retrieve_context("What was the current ratio in FY2025?", "abc")
    assert "FISCAL_PERIOD_MISMATCH" in result.warnings
    assert "FISCAL_YEAR_END_MISMATCH" in result.warnings
    assert "METRIC_DEFINITION_MISMATCH" in result.warnings


def test_missing_fact_returns_insufficient_evidence(store: SQLiteFinancialStore) -> None:
    result = RetrievalGroundingAgent(store).retrieve_context("What was EBITDA in FY2025?", "abc")
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.insufficient_evidence.requested_item == "EBITDA"
    assert result.insufficient_evidence.reason_code == "REQUIRED_FACTS_MISSING"
    assert result.facts == []


def test_faiss_failure_preserves_structured_evidence(store: SQLiteFinancialStore) -> None:
    class UnavailableNarrativeStore:
        def search(self, *args, **kwargs):
            raise NarrativeStoreUnavailableError("index unavailable")

    store.add_fact(make_fact("abc", "revenue", "FY2025", "120"))
    result = RetrievalGroundingAgent(
        store, narrative_store=UnavailableNarrativeStore()
    ).retrieve_context("Why is revenue reported?", "abc")

    assert result.status == "OK"
    assert result.facts[0].value == Decimal("120")
    assert "NARRATIVE_RETRIEVAL_UNAVAILABLE" in result.warnings


def test_trends_and_flags_are_loaded_from_structured_provider(store: SQLiteFinancialStore) -> None:
    store.add_fact(make_fact("abc", "operating_profit", "FY2024", "20"))
    store.add_fact(make_fact("abc", "revenue", "FY2024", "100"))
    store.add_fact(make_fact("abc", "operating_profit", "FY2025", "10"))
    store.add_fact(make_fact("abc", "revenue", "FY2025", "100"))
    store.add_trend(TrendResult(
        trend_id="t1", company_id="abc", metric_name="operating_margin",
        from_period="FY2024", to_period="FY2025", from_value="0.2", to_value="0.1",
        delta="-0.1", supporting_refs=["m1", "m2"],
    ))
    store.add_red_flag(RedFlagResult(
        flag_id="f1", company_id="abc", flag_type="margin_compression", status="TRIGGERED",
        rule_version="margin_compression_v1", period="FY2025", supporting_refs=["t1"],
    ))
    result = RetrievalGroundingAgent(store).retrieve_context("Are there any financial red flags?", "abc", "FY2025")
    assert result.trends[0].trend_id == "t1"
    assert result.red_flags[0].flag_id == "f1"


def test_trend_question_retrieves_prior_period_facts(store: SQLiteFinancialStore) -> None:
    store.add_fact(make_fact("abc", "operating_profit", "FY2024", "20"))
    store.add_fact(make_fact("abc", "revenue", "FY2024", "100"))
    store.add_fact(make_fact("abc", "operating_profit", "FY2025", "10"))
    store.add_fact(make_fact("abc", "revenue", "FY2025", "100"))
    store.add_trend(TrendResult(
        trend_id="t1", company_id="abc", metric_name="operating_margin",
        from_period="FY2024", to_period="FY2025", from_value="0.2", to_value="0.1",
        delta="-0.1", supporting_refs=["m1", "m2"],
    ))

    result = RetrievalGroundingAgent(store).retrieve_context(
        "How did operating margin change in FY2025?", "abc",
    )
    assert set(result.periods) == {"FY2024", "FY2025"}
    assert {fact.fiscal_year for fact in result.facts} == {"FY2024", "FY2025"}
    assert [trend.trend_id for trend in result.trends] == ["t1"]


def test_broad_trend_question_calculates_supported_ratio_trends(
    store: SQLiteFinancialStore,
) -> None:
    store.add_fact(
        make_fact("abc", "current_assets", "FY2024", "150", statement="balance_sheet")
    )
    store.add_fact(
        make_fact("abc", "current_liabilities", "FY2024", "100", statement="balance_sheet")
    )
    store.add_fact(
        make_fact("abc", "current_assets", "FY2025", "200", statement="balance_sheet")
    )
    store.add_fact(
        make_fact("abc", "current_liabilities", "FY2025", "100", statement="balance_sheet")
    )

    result = RetrievalGroundingAgent(store).retrieve_context(
        "What are the financial trends?", "abc",
    )

    current_ratio_trend = next(
        trend for trend in result.trends
        if trend.metric_name == "current_ratio"
    )
    assert result.status == "OK"
    assert current_ratio_trend.quality_status == "VALID"
    assert current_ratio_trend.from_period == "FY2024"
    assert current_ratio_trend.to_period == "FY2025"
    assert current_ratio_trend.from_value == Decimal("1.5000")
    assert current_ratio_trend.to_value == Decimal("2.0000")
    assert len(current_ratio_trend.supporting_refs) == 2
    referenced_metrics = {
        metric.metric_id: metric for metric in result.computed_metrics
    }
    assert all(ref in referenced_metrics for ref in current_ratio_trend.supporting_refs)
    assert all(
        fact_id in {fact.fact_id for fact in result.facts}
        for ref in current_ratio_trend.supporting_refs
        for fact_id in referenced_metrics[ref].input_fact_refs
    )


def test_trend_question_without_two_periods_is_insufficient(
    store: SQLiteFinancialStore,
) -> None:
    store.add_fact(
        make_fact("abc", "current_assets", "FY2025", "200", statement="balance_sheet")
    )
    store.add_fact(
        make_fact("abc", "current_liabilities", "FY2025", "100", statement="balance_sheet")
    )

    result = RetrievalGroundingAgent(store).retrieve_context(
        "What are the financial trends?", "abc",
    )

    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.trends == []


class DeterministicTestEmbedder:
    """Small deterministic vectors keep FAISS tests local and model-free."""

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        vocabulary = ("receivables", "payment", "terms", "cost", "profitability", "inventory")
        return np.asarray(
            [[float(text.casefold().count(term)) for term in vocabulary] for text in texts],
            dtype=np.float32,
        )


def test_narrative_search_filters_before_ranking_and_preserves_provenance() -> None:
    vector_store = FaissNarrativeStore(embedder=DeterministicTestEmbedder())
    vector_store.add_chunks([
        DocumentChunk(
            chunk_id="abc-2025", company_id="abc", company_name="ABC Ltd", document_type="annual_report",
            section="Working Capital", period="FY2025", source_id="s1", source_file="abc.pdf", page=8,
            text="Receivables rose due to longer customer payment terms.", metadata={"topic": "collections"},
        ),
        DocumentChunk(
            chunk_id="xyz-2025", company_id="xyz", company_name="XYZ Ltd", document_type="annual_report",
            section="Working Capital", period="FY2025", source_id="s2", source_file="xyz.pdf", page=9,
            text="Receivables rose sharply.",
        ),
    ])
    results = vector_store.search("receivables payment terms", "abc", "FY2025", top_k=5)
    assert [chunk.chunk_id for chunk in results] == ["abc-2025"]
    assert results[0].metadata == {"topic": "collections"}
    assert results[0].source_reference.citation == "[Annual Report | FY2025 | Working Capital | p.8]"


def test_faiss_store_persists_chunks_and_filters_document_type(tmp_path) -> None:
    vector_store = FaissNarrativeStore(tmp_path, embedder=DeterministicTestEmbedder())
    vector_store.add_chunks([
        DocumentChunk(
            chunk_id="abc-annual", company_id="abc", company_name="ABC Ltd",
            document_type="annual_report", section="Costs", period="FY2025",
            source_id="s1", source_file="annual.pdf", text="Higher input costs reduced profitability.",
        ),
        DocumentChunk(
            chunk_id="abc-policy", company_id="abc", company_name="ABC Ltd",
            document_type="accounting_policy", section="Costs", period="FY2025",
            source_id="s2", source_file="policy.pdf", text="Higher costs are recognized as incurred.",
        ),
    ])

    reloaded = FaissNarrativeStore(tmp_path, embedder=DeterministicTestEmbedder())
    results = reloaded.search("input costs profitability", "abc", "FY2025", "annual_report")
    assert [chunk.chunk_id for chunk in results] == ["abc-annual"]

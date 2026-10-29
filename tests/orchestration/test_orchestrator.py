from __future__ import annotations

from datetime import date
from decimal import Decimal

from agents.narrative_qa.agent import NarrativeQAAgent
from agents.retrieval_grounding.agent import RetrievalGroundingAgent
from models.schemas import FinancialFact
from orchestration import (
    OrchestrationRequest,
    Orchestrator,
    QuestionIntent,
    QuestionRouter,
)
from retrieval.structured_store import SQLiteFinancialStore


def _fact(
    company: str,
    metric: str,
    year: int,
    value: str,
    *,
    statement_type: str = "income_statement",
) -> FinancialFact:
    return FinancialFact(
        fact_id=f"{company}-{metric}-FY{year}",
        company_id=company,
        company_name=company.upper(),
        statement_type=statement_type,
        canonical_line_item=metric,
        period_start=date(year, 1, 1),
        period_end=date(year, 12, 31),
        fiscal_year=f"FY{year}",
        fiscal_year_end=date(year, 12, 31),
        value=Decimal(value),
        currency="GBP",
        unit="absolute",
        source_id=f"source-{company}-FY{year}",
        source_file=f"{company}-FY{year}.pdf",
        locator={"page": 4, "label": metric.replace("_", " ").title()},
        extraction_confidence=0.99,
    )


def _store_with_facts(*facts: FinancialFact) -> SQLiteFinancialStore:
    store = SQLiteFinancialStore(":memory:")
    for fact in facts:
        store.add_fact(fact)
    return store


def test_router_classifies_supported_question_intents() -> None:
    router = QuestionRouter()

    assert router.route("What was revenue in FY2025?").intent == QuestionIntent.EXACT_LOOKUP
    assert router.route("What was the current ratio?").intent == QuestionIntent.RATIO
    assert router.route("How did operating margin change?").intent == QuestionIntent.TREND
    assert router.route("Are there financial warning signs?").intent == QuestionIntent.RED_FLAG
    assert router.route("Why did revenue decline?").intent == QuestionIntent.WHY
    assert router.route("Give me a summary of performance.").intent == QuestionIntent.SUMMARY
    assert router.route(
        "Which company has a stronger liquidity position?",
        has_comparison_company=True,
    ).intent == QuestionIntent.COMPARISON


def test_exact_lookup_runs_retrieval_and_returns_grounded_citation() -> None:
    store = _store_with_facts(_fact("abc", "revenue", 2025, "12000000"))
    response = Orchestrator(RetrievalGroundingAgent(store)).answer(
        OrchestrationRequest(
            question="What was revenue in FY2025?",
            company_id="abc",
        )
    )

    assert response.intent == QuestionIntent.EXACT_LOOKUP
    assert response.status == "OK"
    assert "12,000,000" in response.answer
    assert "[Income Statement | FY2025 | Revenue | p.4]" in response.answer
    assert response.evidence[0].facts[0].fact_id == "abc-revenue-FY2025"
    assert response.citation_validation == "VALID"


def test_trend_question_runs_deterministic_trend_agent() -> None:
    store = _store_with_facts(
        _fact("abc", "revenue", 2024, "100"),
        _fact("abc", "operating_profit", 2024, "20"),
        _fact("abc", "revenue", 2025, "100"),
        _fact("abc", "operating_profit", 2025, "10"),
    )
    response = Orchestrator(RetrievalGroundingAgent(store)).answer(
        OrchestrationRequest(
            question="How did operating margin change?",
            company_id="abc",
        )
    )

    assert response.intent == QuestionIntent.TREND
    trend = next(
        item
        for item in response.evidence[0].trends
        if item.metric_name == "operating_margin"
    )
    assert trend.from_period == "FY2024"
    assert trend.to_period == "FY2025"
    assert trend.delta == Decimal("-0.1000")
    assert "[Income Statement | FY2024 | Operating Profit | p.4]" in response.answer


def test_red_flag_question_retrieves_supported_facts_and_returns_all_detectors() -> None:
    store = _store_with_facts(
        _fact("abc", "revenue", 2024, "100"),
        _fact("abc", "net_income", 2024, "30"),
        _fact(
            "abc",
            "operating_cash_flow",
            2024,
            "25",
            statement_type="cash_flow_statement",
        ),
        _fact(
            "abc",
            "inventory",
            2024,
            "10",
            statement_type="balance_sheet",
        ),
        _fact("abc", "revenue", 2025, "120"),
        _fact("abc", "net_income", 2025, "35"),
        _fact(
            "abc",
            "operating_cash_flow",
            2025,
            "20",
            statement_type="cash_flow_statement",
        ),
        _fact(
            "abc",
            "inventory",
            2025,
            "20",
            statement_type="balance_sheet",
        ),
    )
    response = Orchestrator(RetrievalGroundingAgent(store)).answer(
        OrchestrationRequest(
            question="Are there financial warning signs?",
            company_id="abc",
        )
    )

    assert response.intent == QuestionIntent.RED_FLAG
    assert len(response.evidence[0].red_flags) == 6
    assert {fact.canonical_line_item for fact in response.evidence[0].facts} >= {
        "revenue",
        "net_income",
        "operating_cash_flow",
        "inventory",
    }


def test_comparison_retrieves_independent_evidence_and_checks_citations() -> None:
    store = _store_with_facts(
        _fact(
            "abc",
            "current_assets",
            2025,
            "150",
            statement_type="balance_sheet",
        ),
        _fact(
            "abc",
            "current_liabilities",
            2025,
            "100",
            statement_type="balance_sheet",
        ),
        _fact(
            "xyz",
            "current_assets",
            2025,
            "180",
            statement_type="balance_sheet",
        ),
        _fact(
            "xyz",
            "current_liabilities",
            2025,
            "100",
            statement_type="balance_sheet",
        ),
    )
    response = Orchestrator(RetrievalGroundingAgent(store)).answer(
        OrchestrationRequest(
            question="Compare the current ratio in FY2025.",
            company_id="abc",
            comparison_company_id="xyz",
        )
    )

    assert response.intent == QuestionIntent.COMPARISON
    assert response.status == "OK"
    assert [bundle.company_id for bundle in response.evidence] == ["abc", "xyz"]
    assert response.comparison.metrics[0].comparison_status == "COMPARABLE"
    assert "ABC" in response.answer
    assert "XYZ" in response.answer
    assert "[Balance Sheet | FY2025 | Current Assets | p.4]" in response.answer
    assert response.citation_validation == "VALID"


def test_broad_comparison_expands_known_topics_to_deterministic_metrics() -> None:
    store = _store_with_facts(
        _fact(
            "abc",
            "current_assets",
            2025,
            "150",
            statement_type="balance_sheet",
        ),
        _fact(
            "abc",
            "current_liabilities",
            2025,
            "100",
            statement_type="balance_sheet",
        ),
        _fact(
            "xyz",
            "current_assets",
            2025,
            "180",
            statement_type="balance_sheet",
        ),
        _fact(
            "xyz",
            "current_liabilities",
            2025,
            "100",
            statement_type="balance_sheet",
        ),
    )
    response = Orchestrator(RetrievalGroundingAgent(store)).answer(
        OrchestrationRequest(
            question="Which company has stronger liquidity on the balance sheet?",
            company_id="abc",
            comparison_company_id="xyz",
            period="FY2025",
        )
    )

    compared = {
        metric.metric_name: metric
        for metric in response.comparison.metrics
    }
    assert response.status == "OK_WITH_WARNINGS"
    assert compared["current_ratio"].comparison_status == "COMPARABLE"
    assert compared["current_ratio"].company_a.value == Decimal("1.5000")
    assert compared["current_ratio"].company_b.value == Decimal("1.8000")
    assert response.evidence[0].facts
    assert response.evidence[1].facts


class _NeverGenerate:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return "unsupported"


def test_comparison_with_missing_company_evidence_refuses_without_generation() -> None:
    store = _store_with_facts(
        _fact(
            "abc",
            "current_assets",
            2025,
            "150",
            statement_type="balance_sheet",
        ),
        _fact(
            "abc",
            "current_liabilities",
            2025,
            "100",
            statement_type="balance_sheet",
        ),
    )
    provider = _NeverGenerate()
    narrative = NarrativeQAAgent(llm_client=provider)
    response = Orchestrator(
        RetrievalGroundingAgent(store),
        narrative_agent=narrative,
    ).answer(
        OrchestrationRequest(
            question="Compare the current ratio in FY2025.",
            company_id="abc",
            comparison_company_id="missing",
        )
    )

    assert response.status == "INSUFFICIENT_EVIDENCE"
    assert response.evidence[1].status == "INSUFFICIENT_EVIDENCE"
    assert "cannot be established" in response.answer
    assert provider.calls == 0

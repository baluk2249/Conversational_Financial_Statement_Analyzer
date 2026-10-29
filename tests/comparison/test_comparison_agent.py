from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from agents.comparison import ComparisonAgent, ComparisonRequest
from models.schemas import AnswerEvidence, ComputedMetric, FinancialFact, TrendResult


def _fact(
    company: str,
    metric: str,
    value: str,
    *,
    period: str = "FY2025",
    currency: str | None = "GBP",
    unit: str = "absolute",
    year_end: date = date(2025, 12, 31),
) -> FinancialFact:
    return FinancialFact(
        fact_id=f"{company}-{metric}-{period}",
        company_id=company,
        company_name=company.upper(),
        statement_type="income_statement",
        canonical_line_item=metric,
        metric_definition=f"reported_{metric}",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        fiscal_year=period,
        fiscal_year_end=year_end,
        value=Decimal(value),
        currency=currency,
        unit=unit,
        source_id=f"source-{company}",
        source_file=f"{company}.pdf",
        locator={"page": 4, "label": metric},
        extraction_confidence=0.99,
    )


def _metric(
    company: str,
    name: str,
    value: str,
    *,
    formula_version: str | None = None,
    currency: str | None = "GBP",
    period: str = "FY2025",
    period_start: date = date(2025, 1, 1),
    period_end: date = date(2025, 12, 31),
    year_end: date = date(2025, 12, 31),
) -> ComputedMetric:
    return ComputedMetric(
        metric_id=f"{company}-{name}-{period}",
        company_id=company,
        metric_name=name,
        period=period,
        value=Decimal(value),
        display_unit="ratio",
        formula_version=formula_version or f"{name}_v1",
        input_fact_refs=[f"{company}-{name}-{period}-input"],
        quality_status="VALID",
        currency=currency,
        unit="absolute",
        period_start=period_start,
        period_end=period_end,
        fiscal_year_end=year_end,
    )


def _bundle(
    company: str,
    *,
    metrics: list[ComputedMetric] | None = None,
    facts: list[FinancialFact] | None = None,
    periods: list[str] | None = None,
) -> AnswerEvidence:
    included_facts = list(facts or [])
    included_fact_ids = {fact.fact_id for fact in included_facts}
    for metric in metrics or []:
        if metric.company_id != company:
            continue
        for reference in metric.input_fact_refs:
            if reference in included_fact_ids:
                continue
            start = metric.period_start or date(2025, 1, 1)
            end = metric.period_end or date(2025, 12, 31)
            included_facts.append(
                FinancialFact(
                    fact_id=reference,
                    company_id=company,
                    company_name=company.upper(),
                    statement_type="balance_sheet",
                    canonical_line_item=f"{metric.metric_name}_input",
                    metric_definition=f"{metric.metric_name}_input",
                    period_start=start,
                    period_end=end,
                    fiscal_year=metric.period,
                    fiscal_year_end=metric.fiscal_year_end or end,
                    value=Decimal("1"),
                    currency=metric.currency,
                    unit=metric.unit or "absolute",
                    source_id=f"source-{reference}",
                    source_file=f"{company}.pdf",
                    locator={"page": 1, "label": metric.metric_name},
                    extraction_confidence=0.99,
                )
            )
            included_fact_ids.add(reference)
    return AnswerEvidence(
        question="Compare financial evidence.",
        company_id=company,
        company_name=company.upper(),
        periods=periods or ["FY2025"],
        computed_metrics=metrics or [],
        facts=included_facts,
    )


def _compare(a: AnswerEvidence, b: AnswerEvidence, *names: str):
    return ComparisonAgent().compare(
        a,
        b,
        ComparisonRequest(question="Compare these companies.", metric_names=list(names)),
    )


def test_fully_compatible_metrics_are_compared_deterministically() -> None:
    result = _compare(
        _bundle("abc", metrics=[_metric("abc", "current_ratio", "1.30")]),
        _bundle("xyz", metrics=[_metric("xyz", "current_ratio", "1.85")]),
        "current_ratio",
    )
    item = result.metrics[0]
    assert item.comparison_status == "COMPARABLE"
    assert item.difference == Decimal("0.55")
    assert item.higher_company_id == "xyz"
    assert result.status == "OK"


def test_ratio_remains_comparable_across_currencies_but_surfaces_warning() -> None:
    result = _compare(
        _bundle("abc", metrics=[_metric("abc", "current_ratio", "1.30", currency="GBP")]),
        _bundle("xyz", metrics=[_metric("xyz", "current_ratio", "1.85", currency="USD")]),
        "current_ratio",
    )
    assert result.metrics[0].comparison_status == "COMPARABLE"
    assert result.metrics[0].difference == Decimal("0.55")
    assert "CURRENCY_MISMATCH" in {warning.code for warning in result.warnings}


def test_absolute_values_with_currency_mismatch_are_not_compared() -> None:
    result = _compare(
        _bundle("abc", facts=[_fact("abc", "revenue", "12000000", currency="GBP")]),
        _bundle("xyz", facts=[_fact("xyz", "revenue", "15000000", currency="USD")]),
        "revenue",
    )
    assert result.metrics[0].comparison_status == "INCOMPARABLE"
    assert result.metrics[0].difference is None
    assert result.metrics[0].higher_company_id is None
    assert "CURRENCY_MISMATCH" in {warning.code for warning in result.warnings}


def test_known_unit_scales_are_normalized_without_changing_original_values() -> None:
    result = _compare(
        _bundle("abc", facts=[_fact("abc", "revenue", "12", unit="millions")]),
        _bundle("xyz", facts=[_fact("xyz", "revenue", "15000", unit="thousands")]),
        "revenue",
    )
    item = result.metrics[0]
    assert item.comparison_status == "COMPARABLE"
    assert item.company_a.value == Decimal("12")
    assert item.company_a.unit == "millions"
    assert item.company_a.normalized_value == Decimal("12000000")
    assert item.company_b.normalized_value == Decimal("15000000")
    assert item.difference == Decimal("3000000")
    assert "UNIT_MISMATCH" in item.warning_codes


def test_fiscal_period_mismatch_prevents_comparison() -> None:
    result = _compare(
        _bundle("abc", metrics=[_metric("abc", "current_ratio", "1.3", period="FY2025")]),
        _bundle("xyz", metrics=[_metric("xyz", "current_ratio", "1.8", period="FY2024")]),
        "current_ratio",
    )
    assert result.metrics[0].comparison_status == "INCOMPARABLE"
    assert result.metrics[0].difference is None
    assert "FISCAL_PERIOD_MISMATCH" in {warning.code for warning in result.warnings}


def test_fiscal_year_end_mismatch_prevents_comparison() -> None:
    result = _compare(
        _bundle("abc", metrics=[_metric("abc", "current_ratio", "1.3")]),
        _bundle(
            "xyz",
            metrics=[
                _metric(
                    "xyz",
                    "current_ratio",
                    "1.8",
                    period_end=date(2025, 3, 31),
                    year_end=date(2025, 3, 31),
                )
            ],
        ),
        "current_ratio",
    )
    assert result.metrics[0].comparison_status == "INCOMPARABLE"
    assert "FISCAL_YEAR_END_MISMATCH" in {warning.code for warning in result.warnings}


def test_formula_version_mismatch_prevents_comparison() -> None:
    result = _compare(
        _bundle(
            "abc",
            metrics=[_metric("abc", "operating_margin", "0.15", formula_version="operating_margin_v1")],
        ),
        _bundle(
            "xyz",
            metrics=[_metric("xyz", "operating_margin", "0.18", formula_version="operating_margin_v2")],
        ),
        "operating_margin",
    )
    assert result.metrics[0].comparison_status == "INCOMPARABLE"
    assert "FORMULA_VERSION_MISMATCH" in {warning.code for warning in result.warnings}


def test_missing_metric_definition_prevents_reported_value_comparison() -> None:
    fact_a = _fact("abc", "revenue", "100")
    fact_b = _fact("xyz", "revenue", "200").model_copy(
        update={"metric_definition": None}
    )
    result = _compare(
        _bundle("abc", facts=[fact_a]),
        _bundle("xyz", facts=[fact_b]),
        "revenue",
    )
    assert result.metrics[0].comparison_status == "INCOMPARABLE"
    assert result.metrics[0].difference is None
    assert "METRIC_DEFINITION_MISMATCH" in {warning.code for warning in result.warnings}


@pytest.mark.parametrize(
    ("available_company", "missing_company"),
    [("xyz", "abc"), ("abc", "xyz")],
)
def test_missing_metric_never_produces_a_winner(
    available_company: str, missing_company: str
) -> None:
    available = _bundle(
        available_company,
        metrics=[_metric(available_company, "current_ratio", "1.3")],
    )
    missing = _bundle(missing_company)
    a, b = (available, missing) if available_company == "abc" else (missing, available)
    result = _compare(a, b, "current_ratio")
    item = result.metrics[0]
    assert item.comparison_status == "INSUFFICIENT_EVIDENCE"
    assert item.difference is None
    assert item.higher_company_id is None
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert "MISSING_COMPARISON_EVIDENCE" in {warning.code for warning in result.warnings}


def test_multiple_metrics_are_kept_separate() -> None:
    result = _compare(
        _bundle(
            "abc",
            metrics=[
                _metric("abc", "current_ratio", "1.3"),
                _metric("abc", "debt_to_equity", "0.7"),
            ],
        ),
        _bundle(
            "xyz",
            metrics=[
                _metric("xyz", "current_ratio", "1.8"),
                _metric("xyz", "debt_to_equity", "1.1"),
            ],
        ),
        "current_ratio",
        "debt_to_equity",
    )
    assert [item.metric_name for item in result.metrics] == [
        "current_ratio",
        "debt_to_equity",
    ]
    assert all(item.comparison_status == "COMPARABLE" for item in result.metrics)


def test_company_isolation_ignores_foreign_metric_and_surfaces_warning() -> None:
    result = _compare(
        _bundle(
            "abc",
            metrics=[
                _metric("abc", "current_ratio", "1.3"),
                _metric("xyz", "current_ratio", "99"),
            ],
        ),
        _bundle("xyz", metrics=[_metric("xyz", "current_ratio", "1.8")]),
        "current_ratio",
    )
    item = result.metrics[0]
    assert item.company_a.value == Decimal("1.3")
    assert item.company_a.evidence_refs[0] == "abc-current_ratio-FY2025"
    assert "COMPANY_IDENTITY_MISMATCH" in {warning.code for warning in result.warnings}


def test_comparison_preserves_metric_and_source_provenance() -> None:
    fact_a = _fact("abc", "current_assets", "150")
    fact_b = _fact("xyz", "current_assets", "170")
    metric_a = _metric("abc", "current_ratio", "1.5")
    metric_b = _metric("xyz", "current_ratio", "1.7")
    evidence_a = _bundle("abc", metrics=[metric_a], facts=[fact_a])
    evidence_b = _bundle("xyz", metrics=[metric_b], facts=[fact_b])

    # Point each calculation at the in-bundle source fact.
    evidence_a.computed_metrics[0].input_fact_refs = [fact_a.fact_id]
    evidence_b.computed_metrics[0].input_fact_refs = [fact_b.fact_id]
    result = _compare(evidence_a, evidence_b, "current_ratio")

    assert result.metrics[0].company_a.evidence_refs == [metric_a.metric_id, fact_a.fact_id]
    assert result.metrics[0].company_a.source_references[0].source_id == fact_a.source_id
    assert result.metrics[0].company_b.source_references[0].source_id == fact_b.source_id


def test_missing_metric_source_fact_prevents_unsupported_comparison() -> None:
    metric_a = _metric("abc", "current_ratio", "1.5")
    metric_b = _metric("xyz", "current_ratio", "1.7")
    result = _compare(
        _bundle("abc", metrics=[metric_a]).model_copy(update={"facts": []}),
        _bundle("xyz", metrics=[metric_b]),
        "current_ratio",
    )
    assert result.metrics[0].comparison_status == "INSUFFICIENT_EVIDENCE"
    assert result.metrics[0].difference is None
    assert result.metrics[0].company_a is None
    assert "MISSING_COMPARISON_EVIDENCE" in {warning.code for warning in result.warnings}


def test_improvement_question_compares_grounded_trend_deltas() -> None:
    def trend(company: str, prior: str, current: str) -> TrendResult:
        return TrendResult(
            trend_id=f"{company}-margin-trend",
            company_id=company,
            metric_name="operating_margin",
            from_period="FY2024",
            to_period="FY2025",
            from_value=Decimal(prior),
            to_value=Decimal(current),
            delta=Decimal(current) - Decimal(prior),
            supporting_refs=[
                f"{company}-operating_margin-FY2024",
                f"{company}-operating_margin-FY2025",
            ],
        )

    def history(company: str, before: str, after: str) -> AnswerEvidence:
        return _bundle(
            company,
            periods=["FY2024", "FY2025"],
            metrics=[
                _metric(
                    company,
                    "operating_margin",
                    before,
                    period="FY2024",
                    period_start=date(2024, 1, 1),
                    period_end=date(2024, 12, 31),
                    year_end=date(2024, 12, 31),
                ),
                _metric(
                    company,
                    "operating_margin",
                    after,
                    period="FY2025",
                ),
            ],
        )

    company_a = history("abc", "0.10", "0.12")
    company_b = history("xyz", "0.10", "0.16")
    company_a.trends = [trend("abc", "0.10", "0.12")]
    company_b.trends = [trend("xyz", "0.10", "0.16")]
    result = ComparisonAgent().compare(
        company_a,
        company_b,
        ComparisonRequest(
            question="Which company's margins improved more?",
            metric_names=["operating_margin"],
        ),
    )
    item = result.metrics[0]
    assert item.comparison_status == "COMPARABLE"
    assert item.company_a.value == Decimal("0.02")
    assert item.company_b.value == Decimal("0.06")
    assert item.difference == Decimal("0.04")
    assert item.higher_company_id == "xyz"
    assert "abc-margin-trend" in item.company_a.evidence_refs

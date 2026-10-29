from agents.trend_red_flag.agent import TrendRedFlagAgent
from agents.trend_red_flag.rules import load_rules
from agents.trend_red_flag.trends import analyse_trend


def test_agent_scopes_company_and_preserves_input_evidence(fact_factory):
    facts = [
        fact_factory("revenue", "FY2024", "100"),
        fact_factory("operating_profit", "FY2024", "20"),
        fact_factory("revenue", "FY2025", "100"),
        fact_factory("operating_profit", "FY2025", "10"),
        fact_factory("revenue", "FY2025", "999", company="other"),
    ]
    result = TrendRedFlagAgent(load_rules()).analyse(
        "abc", facts, ["FY2024", "FY2025"]
    )
    operating_margin = next(
        trend for trend in result.trends if trend.metric_name == "operating_margin"
    )
    assert operating_margin.quality_status == "VALID"
    assert {
        metric_id
        for metric_id in operating_margin.supporting_refs
    } == {
        metric.metric_id
        for metric in result.computed_metrics
        if metric.metric_name == "operating_margin"
    }
    assert all(fact.company_id == "abc" for fact in result.facts)
    traceable_refs = (
        {fact.fact_id for fact in result.facts}
        | {metric.metric_id for metric in result.computed_metrics}
        | {trend.trend_id for trend in result.trends}
    )
    assert all(
        ref in traceable_refs
        for flag in result.red_flags
        for ref in flag.supporting_refs
    )


def test_agent_requires_explicit_order_when_dates_are_missing(fact_factory):
    facts = [
        fact_factory("revenue", "FY2024", "100").model_copy(
            update={"period_end": None}
        ),
        fact_factory("revenue", "FY2025", "120").model_copy(
            update={"period_end": None}
        ),
    ]
    try:
        TrendRedFlagAgent().analyse("abc", facts)
    except ValueError as error:
        assert "explicit chronological periods" in str(error)
    else:
        raise AssertionError("Missing period metadata should require explicit ordering")


def test_trend_never_compares_another_company_metric(metric_factory):
    result = analyse_trend(
        "abc",
        "current_ratio",
        ["FY2024", "FY2025"],
        [
            metric_factory("current_ratio", "FY2024", "1.5"),
            metric_factory("current_ratio", "FY2025", "1.2", company="other"),
        ],
    )
    assert result.quality_status == "INSUFFICIENT_EVIDENCE"

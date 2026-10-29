from decimal import Decimal

from agents.trend_red_flag.trends import analyse_trend, analyse_trends


def test_trend_delta_percent_and_provenance(metric_factory):
    metrics = [
        metric_factory("operating_margin", "FY2024", "0.2"),
        metric_factory("operating_margin", "FY2025", "0.15"),
    ]
    trend = analyse_trend("abc", "operating_margin", ["FY2024", "FY2025"], metrics)
    assert trend.delta == Decimal("-0.0500")
    assert trend.absolute_delta == Decimal("-0.0500")
    assert trend.percentage_delta == Decimal("-25.0000")
    assert trend.supporting_refs == [metric.metric_id for metric in metrics]
    assert trend.quality_status == "VALID"


def test_zero_base_has_absolute_but_not_percentage_delta(metric_factory):
    trend = analyse_trend(
        "abc", "dso", ["FY2024", "FY2025"],
        [metric_factory("dso", "FY2024", "0"), metric_factory("dso", "FY2025", "3")],
    )
    assert trend.absolute_delta == Decimal("3.0000")
    assert trend.percentage_delta is None


def test_single_period_and_missing_metric_are_insufficient(metric_factory):
    single_period = analyse_trend(
        "abc", "net_margin", ["FY2025"], [metric_factory("net_margin", "FY2025", "0.1")]
    )
    missing = analyse_trend("abc", "net_margin", ["FY2024", "FY2025"], [])
    assert single_period.quality_status == "INSUFFICIENT_EVIDENCE"
    assert "EXACTLY_TWO_PERIODS_REQUIRED" in single_period.warnings
    assert missing.quality_status == "INSUFFICIENT_EVIDENCE"


def test_incompatible_periods_are_not_compared(metric_factory):
    metrics = [
        metric_factory("current_ratio", "FY2024", "1.5").model_copy(
            update={"unit": "absolute"}
        ),
        metric_factory("current_ratio", "FY2025", "1.2").model_copy(
            update={"unit": "thousands"}
        ),
    ]
    trend = analyse_trend("abc", "current_ratio", ["FY2024", "FY2025"], metrics)
    assert trend.quality_status == "INCOMPATIBLE_INPUTS"
    assert trend.delta is None
    assert "UNIT_MISMATCH" in trend.warnings


def test_analyse_multiple_adjacent_periods(metric_factory):
    metrics = [
        metric_factory("net_margin", "FY2023", "0.2"),
        metric_factory("net_margin", "FY2024", "0.15"),
        metric_factory("net_margin", "FY2025", "0.1"),
    ]
    trends = analyse_trends("abc", "net_margin", ["FY2023", "FY2024", "FY2025"], metrics)
    assert [trend.delta for trend in trends] == [
        Decimal("-0.0500"),
        Decimal("-0.0500"),
    ]

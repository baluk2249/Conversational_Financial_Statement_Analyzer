"""Period-aligned deterministic trend calculations."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from uuid import uuid4

from models.schemas import ComputedMetric, TrendResult

FOUR_PLACES = Decimal("0.0001")


def analyse_trend(
    company_id: str,
    metric_name: str,
    periods: Sequence[str],
    metrics: Sequence[ComputedMetric],
) -> TrendResult:
    """Compare two explicit periods or return a structured unavailable result."""
    if len(periods) != 2:
        return _unavailable(
            company_id,
            metric_name,
            periods[0] if periods else "",
            periods[-1] if periods else "",
            "EXACTLY_TWO_PERIODS_REQUIRED",
        )
    from_period, to_period = periods
    if from_period == to_period:
        return _unavailable(
            company_id, metric_name, from_period, to_period, "PERIODS_MUST_DIFFER"
        )

    from_matches = _matching_metrics(metrics, company_id, metric_name, from_period)
    to_matches = _matching_metrics(metrics, company_id, metric_name, to_period)
    if len(from_matches) != 1 or len(to_matches) != 1:
        return _unavailable(
            company_id,
            metric_name,
            from_period,
            to_period,
            "METRIC_MISSING_OR_AMBIGUOUS",
            _metric_refs(from_matches, to_matches),
        )

    before, after = from_matches[0], to_matches[0]
    refs = [before.metric_id, after.metric_id]
    if before.quality_status != "VALID" or after.quality_status != "VALID" or before.value is None or after.value is None:
        quality_status = (
            "INCOMPATIBLE_INPUTS"
            if "INCOMPATIBLE_INPUTS" in (before.quality_status, after.quality_status)
            else "INSUFFICIENT_EVIDENCE"
        )
        return _unavailable(
            company_id,
            metric_name,
            from_period,
            to_period,
            "METRIC_UNAVAILABLE",
            refs,
            [*before.warnings, *after.warnings],
            quality_status,
        )
    warnings = _compatibility_warnings(before, after)
    if warnings:
        return TrendResult(
            trend_id=f"trend-{uuid4().hex}",
            company_id=company_id,
            metric_name=metric_name,
            from_period=from_period,
            to_period=to_period,
            supporting_refs=refs,
            quality_status="INCOMPATIBLE_INPUTS",
            warnings=warnings,
        )

    delta = (after.value - before.value).quantize(FOUR_PLACES)
    percentage_delta = (
        (delta / before.value * Decimal("100")).quantize(FOUR_PLACES)
        if before.value != 0
        else None
    )
    return TrendResult(
        trend_id=f"trend-{uuid4().hex}",
        company_id=company_id,
        metric_name=metric_name,
        from_period=from_period,
        to_period=to_period,
        from_value=before.value,
        to_value=after.value,
        delta=delta,
        absolute_delta=delta,
        percentage_delta=percentage_delta,
        supporting_refs=refs,
        quality_status="VALID",
    )


def analyse_trends(
    company_id: str,
    metric_name: str,
    periods: Sequence[str],
    metrics: Sequence[ComputedMetric],
) -> list[TrendResult]:
    """Analyze each adjacent pair in a caller-supplied chronological sequence."""
    return [
        analyse_trend(company_id, metric_name, (before, after), metrics)
        for before, after in zip(periods, periods[1:])
    ]


def _matching_metrics(
    metrics: Sequence[ComputedMetric],
    company_id: str,
    metric_name: str,
    period: str,
) -> list[ComputedMetric]:
    return [
        metric for metric in metrics
        if metric.company_id == company_id
        and metric.metric_name == metric_name
        and metric.period == period
    ]


def _metric_refs(*groups: Sequence[ComputedMetric]) -> list[str]:
    return list(dict.fromkeys(metric.metric_id for group in groups for metric in group))


def _compatibility_warnings(before: ComputedMetric, after: ComputedMetric) -> list[str]:
    warnings: list[str] = []
    if before.currency != after.currency:
        warnings.append("CURRENCY_MISMATCH")
    if before.unit != after.unit:
        warnings.append("UNIT_MISMATCH")
    if before.warnings or after.warnings:
        warnings.extend(before.warnings)
        warnings.extend(after.warnings)
    if any(
        metric.period_start and metric.period_end
        and metric.period_end < metric.period_start
        for metric in (before, after)
    ):
        warnings.append("INVALID_PERIOD_RANGE")
    if before.period_start and before.period_end and after.period_start and after.period_end:
        before_days = (before.period_end - before.period_start).days
        after_days = (after.period_end - after.period_start).days
        if abs(before_days - after_days) > 1:
            warnings.append("FISCAL_PERIOD_MISMATCH")
        if before.period_end >= after.period_start:
            warnings.append("OVERLAPPING_OR_REVERSED_PERIODS")
    return list(dict.fromkeys(warnings))


def _unavailable(
    company_id: str,
    metric_name: str,
    from_period: str,
    to_period: str,
    reason: str,
    refs: Sequence[str] = (),
    warnings: Sequence[str] = (),
    quality_status: str = "INSUFFICIENT_EVIDENCE",
) -> TrendResult:
    return TrendResult(
        trend_id=f"trend-{uuid4().hex}",
        company_id=company_id,
        metric_name=metric_name,
        from_period=from_period,
        to_period=to_period,
        supporting_refs=list(refs),
        quality_status=quality_status,
        warnings=list(dict.fromkeys([*warnings, reason])),
    )

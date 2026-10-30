"""Configurable deterministic red-flag detectors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from uuid import uuid4

from agents.trend_red_flag.rules import RuleDefinition, RuleSet
from models.schemas import ComputedMetric, FinancialFact, RedFlagResult, TrendResult

ZERO = Decimal("0")
FLAG_GROUPS = {
    "dso_climbing": {"dso"},
    "margin_compression": {"gross_margin", "operating_margin", "net_margin"},
    "leverage_stress": {"debt_to_equity", "debt_to_assets", "interest_coverage"},
    "liquidity_deterioration": {"current_ratio", "quick_ratio", "operating_cash_flow_ratio"},
}


def evaluate_red_flags(
    company_id: str,
    periods: Sequence[str],
    facts: Sequence[FinancialFact],
    metrics: Sequence[ComputedMetric],
    trends: Sequence[TrendResult],
    rules: RuleSet,
) -> list[RedFlagResult]:
    """Evaluate every configured family for the latest requested period pair."""
    latest = periods[-1] if periods else None
    previous = periods[-2] if len(periods) > 1 else None
    scoped_facts = [
        fact for fact in facts
        if fact.company_id == company_id and (latest is None or fact.fiscal_year == latest)
    ]
    results: list[RedFlagResult] = []
    for flag_type in (
        "accrual_quality_gap", "dso_climbing", "margin_compression",
        "leverage_stress", "inventory_build_up", "liquidity_deterioration",
    ):
        rule = rules.get(flag_type)
        if rule is None:
            results.append(_result(
                company_id, flag_type, "INSUFFICIENT_EVIDENCE", "RULE_NOT_CONFIGURED",
                "UNCONFIGURED",
                latest,
            ))
            continue
        if not rule.enabled:
            results.append(_result(
                company_id, flag_type, "NOT_APPLICABLE", "RULE_DISABLED",
                rule.rule_version, latest,
            ))
            continue

        if flag_type == "accrual_quality_gap":
            result = _accrual_gap(company_id, latest, scoped_facts, rule)
        elif flag_type == "inventory_build_up":
            result = _inventory_build_up(
                company_id, latest, previous, facts, rule
            )
        else:
            result = _trend_flag(
                company_id, flag_type, latest, previous, trends, metrics, rule
            )
        results.append(result)
    return results


def _accrual_gap(
    company_id: str,
    period: str | None,
    facts: Sequence[FinancialFact],
    rule: RuleDefinition,
) -> RedFlagResult:
    threshold = _required_parameter(rule, "minimum_gap_ratio")
    net_income = _one_fact(facts, "net_income")
    operating_cash_flow = _one_fact(facts, "operating_cash_flow")
    if threshold is None:
        return _result(company_id, "accrual_quality_gap", "INSUFFICIENT_EVIDENCE", "THRESHOLD_NOT_CONFIGURED", rule.rule_version, period)
    if net_income is None or operating_cash_flow is None:
        return _result(
            company_id, "accrual_quality_gap", "INSUFFICIENT_EVIDENCE",
            "REQUIRED_EARNINGS_OR_CASH_FLOW_MISSING", rule.rule_version, period,
            _fact_refs(net_income, operating_cash_flow),
        )
    warnings = _fact_compatibility_warnings(net_income, operating_cash_flow)
    if warnings:
        return _result(company_id, "accrual_quality_gap", "INSUFFICIENT_EVIDENCE", warnings[0], rule.rule_version, period, _fact_refs(net_income, operating_cash_flow), warnings)
    scale = max(abs(net_income.value), abs(operating_cash_flow.value))
    if scale == ZERO:
        return _result(
            company_id, "accrual_quality_gap", "NOT_TRIGGERED", "NO_EARNINGS_CASH_FLOW_GAP",
            rule.rule_version, period, _fact_refs(net_income, operating_cash_flow),
        )
    gap_ratio = (net_income.value - operating_cash_flow.value) / scale
    triggered = gap_ratio > threshold
    return _result(
        company_id,
        "accrual_quality_gap",
        "TRIGGERED" if triggered else "NOT_TRIGGERED",
        "EARNINGS_EXCEED_OPERATING_CASH_FLOW" if triggered else "NO_MATERIAL_EARNINGS_CASH_FLOW_GAP",
        rule.rule_version,
        period,
        _fact_refs(net_income, operating_cash_flow),
    )


def _inventory_build_up(
    company_id: str,
    period: str | None,
    previous_period: str | None,
    facts: Sequence[FinancialFact],
    rule: RuleDefinition,
) -> RedFlagResult:
    threshold = _required_parameter(rule, "minimum_growth_gap")
    if threshold is None:
        return _result(company_id, "inventory_build_up", "INSUFFICIENT_EVIDENCE", "THRESHOLD_NOT_CONFIGURED", rule.rule_version, period)
    if period is None or previous_period is None:
        return _result(company_id, "inventory_build_up", "INSUFFICIENT_EVIDENCE", "TWO_PERIODS_REQUIRED", rule.rule_version, period)
    inventory_before = _one_fact(facts, "inventory", previous_period)
    inventory_after = _one_fact(facts, "inventory", period)
    revenue_before = _one_fact(facts, "revenue", previous_period)
    revenue_after = _one_fact(facts, "revenue", period)
    required = (inventory_before, inventory_after, revenue_before, revenue_after)
    if any(fact is None for fact in required):
        return _result(
            company_id, "inventory_build_up", "INSUFFICIENT_EVIDENCE",
            "INVENTORY_OR_REVENUE_MISSING", rule.rule_version, period, _fact_refs(*required),
        )
    before_inventory, after_inventory, before_revenue, after_revenue = required
    assert before_inventory and after_inventory and before_revenue and after_revenue
    warnings = _fact_compatibility_warnings(
        before_inventory, after_inventory, before_revenue, after_revenue,
        require_comparable_periods=True,
    )
    if warnings:
        return _result(company_id, "inventory_build_up", "INSUFFICIENT_EVIDENCE", warnings[0], rule.rule_version, period, _fact_refs(*required), warnings)
    if before_inventory.value == ZERO or before_revenue.value == ZERO:
        return _result(
            company_id, "inventory_build_up", "INSUFFICIENT_EVIDENCE",
            "ZERO_BASELINE_PREVENTS_GROWTH_COMPARISON", rule.rule_version, period,
            _fact_refs(*required),
        )
    inventory_growth = after_inventory.value / before_inventory.value - Decimal("1")
    revenue_growth = after_revenue.value / before_revenue.value - Decimal("1")
    triggered = inventory_growth - revenue_growth > threshold
    return _result(
        company_id,
        "inventory_build_up",
        "TRIGGERED" if triggered else "NOT_TRIGGERED",
        "INVENTORY_GROWTH_EXCEEDS_REVENUE" if triggered else "INVENTORY_GROWTH_ALIGNED_WITH_REVENUE",
        rule.rule_version,
        period,
        _fact_refs(*required),
    )


def _trend_flag(
    company_id: str,
    flag_type: str,
    period: str | None,
    previous_period: str | None,
    trends: Sequence[TrendResult],
    metrics: Sequence[ComputedMetric],
    rule: RuleDefinition,
) -> RedFlagResult:
    if period is None or previous_period is None:
        return _result(company_id, flag_type, "INSUFFICIENT_EVIDENCE", "TWO_PERIODS_REQUIRED", rule.rule_version, period)
    metric_names = FLAG_GROUPS[flag_type]
    relevant = [
        trend for trend in trends
        if trend.company_id == company_id
        and trend.metric_name in metric_names
        and trend.from_period == previous_period
        and trend.to_period == period
    ]
    if not relevant:
        return _result(company_id, flag_type, "INSUFFICIENT_EVIDENCE", "NO_COMPARABLE_TRENDS", rule.rule_version, period)

    missing_parameters = _required_parameters(flag_type, rule)
    if missing_parameters:
        return _result(company_id, flag_type, "INSUFFICIENT_EVIDENCE", "THRESHOLD_NOT_CONFIGURED", rule.rule_version, period, _trend_refs(relevant), [f"MISSING_PARAMETER:{name}" for name in missing_parameters])

    usable = [trend for trend in relevant if trend.quality_status == "VALID" and trend.delta is not None]
    if not usable:
        warnings = [warning for trend in relevant for warning in trend.warnings]
        return _result(company_id, flag_type, "INSUFFICIENT_EVIDENCE", "TREND_UNAVAILABLE_OR_INCOMPATIBLE", rule.rule_version, period, _trend_refs(relevant), warnings)

    triggered_trends: list[TrendResult] = []
    for trend in usable:
        threshold = _threshold_for_metric(flag_type, rule)
        if threshold is None:
            continue
        adverse = (
            trend.delta > threshold if flag_type == "dso_climbing"
            else trend.delta < -threshold if trend.metric_name == "interest_coverage"
            else trend.delta < -threshold if flag_type in {"margin_compression", "liquidity_deterioration"}
            else trend.delta > threshold
        )
        if adverse:
            triggered_trends.append(trend)
    triggered = bool(triggered_trends)
    refs = _trend_refs(triggered_trends if triggered else usable)
    evidence_metric_ids = {
        metric.metric_id
        for metric in metrics
        if metric.company_id == company_id
        and metric.metric_name in metric_names
        and metric.period in {previous_period, period}
    }
    refs = list(dict.fromkeys(refs + sorted(evidence_metric_ids)))
    code = (
        triggered_trends[0].metric_name.upper() + "_DETERIORATION"
        if triggered and flag_type in {"margin_compression", "liquidity_deterioration"}
        else triggered_trends[0].metric_name.upper() + "_INCREASE"
        if triggered and triggered_trends[0].metric_name != "interest_coverage"
        else triggered_trends[0].metric_name.upper() + "_DECLINE"
        if triggered
        else f"NO_{flag_type.upper()}_DETECTED"
    )
    return _result(
        company_id, flag_type, "TRIGGERED" if triggered else "NOT_TRIGGERED",
        code, rule.rule_version, period, refs,
    )


def _required_parameters(flag_type: str, rule: RuleDefinition) -> list[str]:
    if flag_type == "dso_climbing":
        names = ("minimum_increase",)
    elif flag_type == "margin_compression":
        names = ("minimum_decline",)
    elif flag_type == "leverage_stress":
        names = ("minimum_worsening",)
    else:
        names = ("minimum_decline",)
    return [name for name in names if name not in rule.parameters]


def _threshold_for_metric(flag_type: str, rule: RuleDefinition) -> Decimal | None:
    if flag_type == "dso_climbing":
        return rule.parameters.get("minimum_increase")
    if flag_type == "margin_compression":
        return rule.parameters.get("minimum_decline")
    if flag_type == "leverage_stress":
        return rule.parameters.get("minimum_worsening")
    return rule.parameters.get("minimum_decline")


def _required_parameter(rule: RuleDefinition, name: str) -> Decimal | None:
    return rule.parameters.get(name)


def _one_fact(
    facts: Sequence[FinancialFact], metric: str, period: str | None = None
) -> FinancialFact | None:
    matches = [
        fact for fact in facts
        if fact.canonical_line_item == metric
        and (period is None or fact.fiscal_year == period)
    ]
    measurements = {
        (
            fact.value,
            fact.currency,
            fact.unit,
            fact.period_start,
            fact.period_end,
            fact.fiscal_year_end,
            fact.metric_definition,
        ): fact
        for fact in matches
    }
    return next(iter(measurements.values())) if len(measurements) == 1 else None


def _fact_compatibility_warnings(
    *facts: FinancialFact, require_comparable_periods: bool = False
) -> list[str]:
    warnings: list[str] = []
    currencies = {fact.currency for fact in facts}
    units = {fact.unit for fact in facts}
    if len(currencies) != 1:
        warnings.append("CURRENCY_MISMATCH")
    if len(units) != 1:
        warnings.append("UNIT_MISMATCH")
    periods: dict[str, list[FinancialFact]] = defaultdict(list)
    for fact in facts:
        if (
            fact.period_start
            and fact.period_end
            and fact.period_end < fact.period_start
        ):
            warnings.append("INVALID_PERIOD_RANGE")
        periods[fact.fiscal_year].append(fact)
    for same_period_facts in periods.values():
        starts = {fact.period_start for fact in same_period_facts if fact.period_start}
        ends = {fact.period_end for fact in same_period_facts if fact.period_end}
        year_ends = {fact.fiscal_year_end for fact in same_period_facts if fact.fiscal_year_end}
        if len(starts) > 1 or len(ends) > 1:
            warnings.append("FISCAL_PERIOD_MISMATCH")
        if len(year_ends) > 1:
            warnings.append("FISCAL_YEAR_END_MISMATCH")
    if require_comparable_periods:
        for metric in {fact.canonical_line_item for fact in facts}:
            matching = [fact for fact in facts if fact.canonical_line_item == metric]
            if len(matching) == 2:
                before, after = sorted(
                    matching,
                    key=lambda fact: fact.period_end or fact.period_start or date.min,
                )
                if (
                    before.period_start is None
                    or before.period_end is None
                    or after.period_start is None
                    or after.period_end is None
                ):
                    warnings.append("FISCAL_PERIOD_METADATA_MISSING")
                    continue
                before_days = (before.period_end - before.period_start).days
                after_days = (after.period_end - after.period_start).days
                if abs(before_days - after_days) > 1:
                    warnings.append("FISCAL_PERIOD_MISMATCH")
                if before.period_end >= after.period_start:
                    warnings.append("OVERLAPPING_OR_REVERSED_PERIODS")
                if (
                    before.fiscal_year_end
                    and after.fiscal_year_end
                    and (before.fiscal_year_end.month, before.fiscal_year_end.day)
                    != (after.fiscal_year_end.month, after.fiscal_year_end.day)
                ):
                    warnings.append("FISCAL_YEAR_END_MISMATCH")
                if before.metric_definition != after.metric_definition:
                    warnings.append("METRIC_DEFINITION_MISMATCH")
    return list(dict.fromkeys(warnings))


def _fact_refs(*facts: FinancialFact | None) -> list[str]:
    return list(dict.fromkeys(fact.fact_id for fact in facts if fact is not None))


def _trend_refs(trends: Sequence[TrendResult]) -> list[str]:
    refs = [trend.trend_id for trend in trends]
    refs.extend(ref for trend in trends for ref in trend.supporting_refs)
    return list(dict.fromkeys(refs))


def _result(
    company_id: str,
    flag_type: str,
    status: str,
    explanation_code: str,
    rule_version: str,
    period: str | None,
    supporting_refs: Sequence[str] = (),
    warnings: Sequence[str] = (),
) -> RedFlagResult:
    return RedFlagResult(
        flag_id=f"flag-{uuid4().hex}",
        company_id=company_id,
        flag_type=flag_type,
        status=status,
        rule_version=rule_version,
        period=period,
        supporting_refs=list(dict.fromkeys(supporting_refs)),
        explanation_code=explanation_code,
        warnings=list(warnings),
    )

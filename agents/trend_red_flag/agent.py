"""Orchestration facade for deterministic trend and red-flag analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from agents.trend_red_flag.red_flags import evaluate_red_flags
from agents.trend_red_flag.rules import RuleSet, load_rules
from agents.trend_red_flag.trends import analyse_trends
from finance.ratios import calculate_ratio
from models.schemas import ComputedMetric, FinancialFact, RedFlagResult, TrendResult

RATIO_NAMES = (
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


@dataclass(frozen=True)
class TrendRedFlagAnalysis:
    company_id: str
    periods: list[str]
    facts: list[FinancialFact]
    computed_metrics: list[ComputedMetric]
    trends: list[TrendResult]
    red_flags: list[RedFlagResult]


class TrendRedFlagAgent:
    """Run the finance engine, adjacent-period trend engine, and configured rules."""

    def __init__(self, rules: RuleSet | None = None) -> None:
        self.rules = rules or load_rules()

    def analyse(
        self,
        company_id: str,
        facts: Sequence[FinancialFact],
        periods: Sequence[str] | None = None,
        computed_metrics: Sequence[ComputedMetric] = (),
    ) -> TrendRedFlagAnalysis:
        """Analyze one company; explicit periods must be supplied chronologically."""
        company_facts = [fact for fact in facts if fact.company_id == company_id]
        company_metrics = [
            metric for metric in computed_metrics if metric.company_id == company_id
        ]
        ordered_periods = list(periods) if periods is not None else _infer_periods(company_facts)
        if len(set(ordered_periods)) != len(ordered_periods):
            raise ValueError("Periods must be unique and chronologically ordered")

        metrics = list(company_metrics)
        existing = {(metric.metric_name, metric.period) for metric in metrics}
        for period in ordered_periods:
            for metric_name in RATIO_NAMES:
                if (metric_name, period) not in existing:
                    metrics.append(
                        calculate_ratio(company_facts, company_id, metric_name, period)
                    )

        trends = [
            trend
            for metric_name in RATIO_NAMES
            for trend in analyse_trends(
                company_id, metric_name, ordered_periods, metrics
            )
        ]
        red_flags = evaluate_red_flags(
            company_id, ordered_periods, company_facts, metrics, trends, self.rules
        )
        return TrendRedFlagAnalysis(
            company_id=company_id,
            periods=ordered_periods,
            facts=company_facts,
            computed_metrics=metrics,
            trends=trends,
            red_flags=red_flags,
        )

    def analyze(
        self,
        company_id: str,
        facts: Sequence[FinancialFact],
        periods: Sequence[str] | None = None,
        computed_metrics: Sequence[ComputedMetric] = (),
    ) -> TrendRedFlagAnalysis:
        """US spelling alias for :meth:`analyse`."""
        return self.analyse(company_id, facts, periods, computed_metrics)


def _infer_periods(facts: Sequence[FinancialFact]) -> list[str]:
    by_period: dict[str, set[date | None]] = {}
    for fact in facts:
        by_period.setdefault(fact.fiscal_year, set()).add(fact.period_end)
    if len(by_period) <= 1:
        return list(by_period)
    if any(len(end_dates) != 1 or None in end_dates for end_dates in by_period.values()):
        raise ValueError(
            "Pass an explicit chronological periods sequence when fiscal-period "
            "end dates are missing or inconsistent"
        )
    period_ends = [next(iter(end_dates)) for end_dates in by_period.values()]
    if len(set(period_ends)) != len(period_ends):
        raise ValueError("Distinct fiscal periods must have distinct period-end dates")
    return sorted(
        by_period,
        key=lambda period: next(iter(by_period[period])),
    )

"""Deterministic ratio calculation over normalized, provenance-bearing facts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal, DivisionByZero, InvalidOperation
from uuid import uuid4

from agents.retrieval_grounding.config import COMPUTED_METRIC_INPUTS
from models.schemas import ComputedMetric, FinancialFact
from retrieval.interfaces import StructuredFinancialStore

ZERO = Decimal("0")
FOUR_PLACES = Decimal("0.0001")
AVERAGE_BALANCE_METRICS = {
    "roa": "total_assets",
    "roe": "equity",
    "dso": "accounts_receivable",
    "inventory_days": "inventory",
    "asset_turnover": "total_assets",
}


def calculate_ratio(
    facts: Sequence[FinancialFact],
    company_id: str,
    metric: str,
    period: str,
) -> ComputedMetric:
    """Calculate one supported ratio from facts, preserving all available refs."""
    if metric not in COMPUTED_METRIC_INPUTS:
        raise ValueError(f"Unsupported computed metric: {metric}")

    required = COMPUTED_METRIC_INPUTS[metric]
    scoped_facts = [fact for fact in facts if fact.company_id == company_id]
    grouped: dict[str, dict[str, list[FinancialFact]]] = defaultdict(lambda: defaultdict(list))
    for fact in scoped_facts:
        grouped[fact.fiscal_year][fact.canonical_line_item].append(fact)

    current: dict[str, FinancialFact] = {}
    warnings: list[str] = []
    for name in required:
        matches = grouped.get(period, {}).get(name, [])
        if len(matches) > 1:
            warnings.append(f"AMBIGUOUS_INPUT:{name}")
        elif matches:
            current[name] = matches[0]

    missing = [name for name in required if name not in current]
    selected = list(current.values())
    prior_period: str | None = None
    average_metric = AVERAGE_BALANCE_METRICS.get(metric)
    if average_metric and average_metric in current:
        prior_period = _find_prior_period(period, current[average_metric], grouped, average_metric)
        if prior_period is None:
            missing.append(f"prior_period:{average_metric}")
        else:
            prior_matches = grouped[prior_period][average_metric]
            if len(prior_matches) != 1:
                warnings.append(f"AMBIGUOUS_INPUT:{average_metric}:{prior_period}")
                missing.append(f"prior_period:{average_metric}")
            else:
                selected.append(prior_matches[0])

    selected = _unique_facts(selected)
    warnings.extend(_compatibility_warnings(selected, period))
    if metric in {"dso", "inventory_days"}:
        activity_fact = current.get("revenue" if metric == "dso" else "cost_of_goods_sold")
        if activity_fact and (activity_fact.period_start is None or activity_fact.period_end is None):
            missing.append("period_start_and_end")

    if missing or warnings:
        return _unavailable(company_id, metric, period, selected, missing, warnings)

    try:
        values = {name: fact.value for name, fact in current.items()}
        value = _formula(metric, values, current, grouped, prior_period)
    except (DivisionByZero, InvalidOperation, ZeroDivisionError):
        return _unavailable(
            company_id, metric, period, selected, ["non_zero_denominator"], []
        )

    return ComputedMetric(
        metric_id=f"calc-{uuid4().hex}",
        company_id=company_id,
        metric_name=metric,
        period=period,
        value=value.quantize(FOUR_PLACES),
        display_unit=(
            "days" if metric in {"dso", "inventory_days"}
            else "percent" if metric in {"gross_margin", "operating_margin", "net_margin", "roa", "roe"}
            else "ratio"
        ),
        formula_version=f"{metric}_v1",
        input_fact_refs=[fact.fact_id for fact in selected],
        quality_status="VALID",
        currency=selected[0].currency if selected else None,
        unit=selected[0].unit if selected else None,
        period_start=current[required[0]].period_start if required[0] in current else None,
        period_end=current[required[0]].period_end if required[0] in current else None,
        fiscal_year_end=current[required[0]].fiscal_year_end if required[0] in current else None,
    )


class RatioCalculator:
    """Store-backed adapter for the reusable fact-based ratio engine."""

    def __init__(self, store: StructuredFinancialStore) -> None:
        self.store = store

    def calculate(self, company_id: str, metric: str, period: str) -> ComputedMetric:
        if metric not in COMPUTED_METRIC_INPUTS:
            raise ValueError(f"Unsupported computed metric: {metric}")
        facts = self.store.get_facts(company_id, COMPUTED_METRIC_INPUTS[metric])
        return calculate_ratio(facts, company_id, metric, period)


def _unique_facts(facts: Sequence[FinancialFact]) -> list[FinancialFact]:
    result: list[FinancialFact] = []
    seen: set[str] = set()
    for fact in facts:
        if fact.fact_id not in seen:
            result.append(fact)
            seen.add(fact.fact_id)
    return result


def _find_prior_period(
    period: str,
    current_fact: FinancialFact,
    grouped: dict[str, dict[str, list[FinancialFact]]],
    metric: str,
) -> str | None:
    if current_fact.period_start is None:
        return None
    candidates = [
        (candidate, matches[0])
        for candidate, line_items in grouped.items()
        if candidate != period
        and len(matches := line_items.get(metric, [])) == 1
        and matches[0].period_end is not None
        and matches[0].period_end < current_fact.period_start
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1].period_end)[0]


def _formula(
    metric: str,
    values: dict[str, Decimal],
    current: dict[str, FinancialFact],
    grouped: dict[str, dict[str, list[FinancialFact]]],
    prior_period: str | None,
) -> Decimal:
    formulas = {
        "current_ratio": lambda: values["current_assets"] / values["current_liabilities"],
        "quick_ratio": lambda: (values["current_assets"] - values["inventory"]) / values["current_liabilities"],
        "operating_cash_flow_ratio": lambda: values["operating_cash_flow"] / values["current_liabilities"],
        "debt_to_equity": lambda: values["total_debt"] / values["equity"],
        "debt_to_assets": lambda: values["total_debt"] / values["total_assets"],
        "interest_coverage": lambda: values["ebit"] / values["interest_expense"],
        "gross_margin": lambda: values["gross_profit"] / values["revenue"],
        "operating_margin": lambda: values["operating_profit"] / values["revenue"],
        "net_margin": lambda: values["net_income"] / values["revenue"],
        "roa": lambda: values["net_income"] / _average("total_assets", current, grouped, prior_period),
        "roe": lambda: values["net_income"] / _average("equity", current, grouped, prior_period),
        "dso": lambda: _average("accounts_receivable", current, grouped, prior_period)
        / values["revenue"] * _days(current["revenue"]),
        "inventory_days": lambda: _average("inventory", current, grouped, prior_period)
        / values["cost_of_goods_sold"] * _days(current["cost_of_goods_sold"]),
        "asset_turnover": lambda: values["revenue"] / _average("total_assets", current, grouped, prior_period),
    }
    return formulas[metric]()


def _average(
    metric: str,
    current: dict[str, FinancialFact],
    grouped: dict[str, dict[str, list[FinancialFact]]],
    prior_period: str | None,
) -> Decimal:
    if prior_period is None:
        raise ValueError(f"A prior-period {metric} fact is required")
    prior = grouped[prior_period][metric][0]
    return (current[metric].value + prior.value) / Decimal("2")


def _days(fact: FinancialFact) -> Decimal:
    if fact.period_start is None or fact.period_end is None:
        raise ValueError("A reporting period is required to calculate days-based ratios")
    return Decimal((fact.period_end - fact.period_start).days + 1)


def _compatibility_warnings(facts: Sequence[FinancialFact], period: str) -> list[str]:
    currencies = {fact.currency for fact in facts if fact.currency}
    units = {fact.unit for fact in facts if fact.unit}
    current_facts = [fact for fact in facts if fact.fiscal_year == period]
    lengths = {
        (fact.period_end - fact.period_start).days
        for fact in current_facts if fact.period_start and fact.period_end
    }
    fiscal_year_ends = {fact.fiscal_year_end for fact in current_facts if fact.fiscal_year_end}
    definitions: dict[str, set[str]] = defaultdict(set)
    for fact in facts:
        if fact.metric_definition:
            definitions[fact.canonical_line_item].add(fact.metric_definition)
    warnings: list[str] = []
    if any(
        fact.period_start and fact.period_end and fact.period_end < fact.period_start
        for fact in facts
    ):
        warnings.append("INVALID_PERIOD_RANGE")
    if len(currencies) > 1:
        warnings.append("CURRENCY_MISMATCH")
    if len(units) > 1:
        warnings.append("UNIT_MISMATCH")
    if lengths and max(lengths) - min(lengths) > 1:
        warnings.append("FISCAL_PERIOD_MISMATCH")
    if len(fiscal_year_ends) > 1:
        warnings.append("FISCAL_YEAR_END_MISMATCH")
    if any(len(values) > 1 for values in definitions.values()):
        warnings.append("METRIC_DEFINITION_MISMATCH")
    return warnings


def _unavailable(
    company_id: str,
    metric: str,
    period: str,
    facts: Sequence[FinancialFact],
    missing: Sequence[str],
    warnings: Sequence[str],
) -> ComputedMetric:
    return ComputedMetric(
        metric_id=f"calc-{uuid4().hex}",
        company_id=company_id,
        metric_name=metric,
        period=period,
        value=None,
        display_unit=(
            "days" if metric in {"dso", "inventory_days"}
            else "percent" if metric in {"gross_margin", "operating_margin", "net_margin", "roa", "roe"}
            else "ratio"
        ),
        formula_version=f"{metric}_v1",
        input_fact_refs=[fact.fact_id for fact in facts],
        quality_status="INCOMPATIBLE_INPUTS" if warnings else "INSUFFICIENT_EVIDENCE",
        warnings=list(warnings) + [f"MISSING_INPUT:{item}" for item in missing],
        currency=facts[0].currency if facts else None,
        unit=facts[0].unit if facts else None,
    )

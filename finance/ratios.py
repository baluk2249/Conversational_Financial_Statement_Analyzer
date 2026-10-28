"""Deterministic ratio calculation over normalized, provenance-bearing facts."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Sequence
from uuid import uuid4

from agents.retrieval_grounding.config import COMPUTED_METRIC_INPUTS
from models.schemas import ComputedMetric, FinancialFact
from retrieval.interfaces import StructuredFinancialStore

logger = logging.getLogger(__name__)
ZERO = Decimal("0")


class RatioCalculator:
    """Computes configured ratios using Decimal and records exact input fact IDs."""

    def __init__(self, store: StructuredFinancialStore) -> None:
        self.store = store

    def calculate(self, company_id: str, metric: str, period: str) -> ComputedMetric:
        if metric not in COMPUTED_METRIC_INPUTS:
            raise ValueError(f"Unsupported computed metric: {metric}")
        required = COMPUTED_METRIC_INPUTS[metric]
        fact_rows = self.store.get_facts(company_id, required)
        by_period: dict[str, dict[str, FinancialFact]] = defaultdict(dict)
        for fact in fact_rows:
            by_period[fact.fiscal_year][fact.canonical_line_item] = fact

        current = by_period.get(period, {})
        missing = [name for name in required if name not in current]
        prior_period = self._prior_period(period, current, by_period)
        uses_average = metric in {"roa", "roe", "dso", "inventory_days", "asset_turnover"}
        if uses_average and prior_period is None:
            balance_metric = self._average_balance_metric(metric)
            if balance_metric in current:
                missing.append(f"prior_period:{balance_metric}")
        elif uses_average and prior_period and self._average_balance_metric(metric) not in by_period[prior_period]:
            missing.append(f"prior_period:{self._average_balance_metric(metric)}")
        if missing:
            return self._unavailable(company_id, metric, period, required, missing)

        selected = [current[name] for name in required]
        if uses_average and prior_period:
            selected.append(by_period[prior_period][self._average_balance_metric(metric)])
        compatibility = self._compatibility_warnings(selected, period)
        if compatibility:
            return self._unavailable(company_id, metric, period, required, [], compatibility)

        values = {name: current[name].value for name in required}
        try:
            value = self._formula(metric, values, current, by_period, prior_period)
        except (DivisionByZero, InvalidOperation, ZeroDivisionError):
            return self._unavailable(company_id, metric, period, required, ["non_zero_denominator"])

        return ComputedMetric(
            metric_id=f"calc-{uuid4().hex}",
            company_id=company_id,
            metric_name=metric,
            period=period,
            value=value,
            display_unit="days" if metric in {"dso", "inventory_days"} else "percent" if "margin" in metric or metric in {"roa", "roe"} else "ratio",
            formula_version=f"{metric}_v1",
            input_fact_refs=[fact.fact_id for fact in selected],
            quality_status="VALID",
            currency=selected[0].currency,
            unit=selected[0].unit,
        )

    @staticmethod
    def _formula(
        metric: str,
        values: dict[str, Decimal],
        current: dict[str, FinancialFact],
        by_period: dict[str, dict[str, FinancialFact]],
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
            "roa": lambda: values["net_income"] / RatioCalculator._average("total_assets", current, by_period, prior_period),
            "roe": lambda: values["net_income"] / RatioCalculator._average("equity", current, by_period, prior_period),
            "dso": lambda: RatioCalculator._average("accounts_receivable", current, by_period, prior_period)
            / values["revenue"] * RatioCalculator._days(current["revenue"]),
            "inventory_days": lambda: RatioCalculator._average("inventory", current, by_period, prior_period)
            / values["cost_of_goods_sold"] * RatioCalculator._days(current["cost_of_goods_sold"]),
            "asset_turnover": lambda: values["revenue"] / RatioCalculator._average("total_assets", current, by_period, prior_period),
        }
        return formulas[metric]().quantize(Decimal("0.0001"))

    @staticmethod
    def _average(metric: str, current: dict[str, FinancialFact], by_period: dict[str, dict[str, FinancialFact]], prior_period: str | None) -> Decimal:
        if prior_period is None:
            raise ValueError(f"A prior-period {metric} fact is required")
        return (current[metric].value + by_period[prior_period][metric].value) / Decimal("2")

    @staticmethod
    def _average_balance_metric(metric: str) -> str:
        return {
            "roa": "total_assets", "roe": "equity", "dso": "accounts_receivable",
            "inventory_days": "inventory", "asset_turnover": "total_assets",
        }[metric]

    @staticmethod
    def _prior_period(
        period: str, current: dict[str, FinancialFact], by_period: dict[str, dict[str, FinancialFact]],
    ) -> str | None:
        current_fact = next(iter(current.values()), None)
        candidates = [candidate for candidate in by_period if candidate != period]
        if current_fact and current_fact.period_start:
            dated = [
                candidate for candidate in candidates
                if (fact := next(iter(by_period[candidate].values()), None))
                and fact.period_end and fact.period_end < current_fact.period_start
            ]
            if dated:
                return max(dated, key=lambda candidate: next(iter(by_period[candidate].values())).period_end or date.min)
        current_year = RatioCalculator._fiscal_year_number(period)
        if current_year is None:
            return None
        earlier = [candidate for candidate in candidates if (year := RatioCalculator._fiscal_year_number(candidate)) is not None and year < current_year]
        return max(earlier, key=RatioCalculator._fiscal_year_number) if earlier else None

    @staticmethod
    def _fiscal_year_number(period: str) -> int | None:
        import re

        match = re.search(r"(\d{4})", period)
        return int(match.group(1)) if match else None

    @staticmethod
    def _days(fact: FinancialFact) -> Decimal:
        if fact.period_start and fact.period_end:
            return Decimal((fact.period_end - fact.period_start).days + 1)
        return Decimal("365")

    @staticmethod
    def _compatibility_warnings(facts: Sequence[FinancialFact], period: str) -> list[str]:
        currencies = {fact.currency for fact in facts if fact.currency}
        units = {fact.unit for fact in facts if fact.unit}
        current_facts = [fact for fact in facts if fact.fiscal_year == period]
        lengths = {
            (fact.period_end - fact.period_start).days + 1
            for fact in current_facts if fact.period_start and fact.period_end
        }
        fiscal_year_ends = {fact.fiscal_year_end for fact in current_facts if fact.fiscal_year_end}
        definitions: dict[str, set[str]] = defaultdict(set)
        for fact in facts:
            if fact.metric_definition:
                definitions[fact.canonical_line_item].add(fact.metric_definition)
        warnings: list[str] = []
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

    @staticmethod
    def _unavailable(
        company_id: str, metric: str, period: str, inputs: Sequence[str],
        missing: Sequence[str], warnings: Sequence[str] = (),
    ) -> ComputedMetric:
        return ComputedMetric(
            metric_id=f"calc-{uuid4().hex}",
            company_id=company_id,
            metric_name=metric,
            period=period,
            value=None,
            display_unit="days" if metric in {"dso", "inventory_days"} else "ratio",
            formula_version=f"{metric}_v1",
            input_fact_refs=[],
            quality_status="INCOMPATIBLE_INPUTS" if warnings else "INSUFFICIENT_EVIDENCE",
            warnings=[*warnings, *(f"MISSING_INPUT:{item}" for item in missing)],
        )
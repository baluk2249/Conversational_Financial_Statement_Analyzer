from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from models.schemas import ComputedMetric, FinancialFact


@pytest.fixture
def fact_factory():
    def make_fact(
        metric: str,
        period: str,
        value: str,
        *,
        company: str = "abc",
        start: date | None = None,
        end: date | None = None,
        currency: str | None = "GBP",
        unit: str = "absolute",
        definition: str | None = None,
        statement: str = "income_statement",
    ) -> FinancialFact:
        year = int(period[-4:])
        return FinancialFact(
            fact_id=f"{company}-{metric}-{period}",
            company_id=company,
            company_name=company.upper(),
            statement_type=statement,
            canonical_line_item=metric,
            metric_definition=definition,
            period_start=start or date(year, 1, 1),
            period_end=end or date(year, 12, 31),
            fiscal_year=period,
            fiscal_year_end=end or date(year, 12, 31),
            value=Decimal(value),
            currency=currency,
            unit=unit,
            source_id=f"source-{company}-{period}",
            source_file=f"{company}-{period}.pdf",
            locator={"page": 1, "label": metric},
            extraction_confidence=0.99,
        )
    return make_fact


@pytest.fixture
def metric_factory():
    def make_metric(
        name: str,
        period: str,
        value: str,
        *,
        company: str = "abc",
        metric_id: str | None = None,
        currency: str | None = "GBP",
        unit: str | None = "absolute",
        quality_status: str = "VALID",
        warnings: list[str] | None = None,
    ) -> ComputedMetric:
        year = int(period[-4:])
        return ComputedMetric(
            metric_id=metric_id or f"{company}-{name}-{period}",
            company_id=company,
            metric_name=name,
            period=period,
            value=Decimal(value) if quality_status == "VALID" else None,
            display_unit="ratio",
            formula_version=f"{name}_v1",
            input_fact_refs=[f"fact-{name}-{period}"],
            quality_status=quality_status,
            currency=currency,
            unit=unit,
            warnings=warnings or [],
            period_start=date(year, 1, 1),
            period_end=date(year, 12, 31),
            fiscal_year_end=date(year, 12, 31),
        )
    return make_metric

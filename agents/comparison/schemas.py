"""Typed contracts for evidence-backed company comparisons."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field

from models.schemas import FinancialModel, SourceReference


class ComparisonRequest(FinancialModel):
    question: str
    metric_names: list[str] = Field(default_factory=list)
    period: str | None = None
    compare_trends: bool = False


class ComparisonWarning(FinancialModel):
    code: str
    message: str
    metric_name: str | None = None
    company_ids: list[str] = Field(default_factory=list)


class ComparisonEvidence(FinancialModel):
    company_id: str
    company_name: str | None = None
    value: Decimal
    period: str
    currency: str | None = None
    unit: str | None = None
    display_unit: str | None = None
    normalized_value: Decimal | None = None
    normalized_unit: str | None = None
    metric_definition: str | None = None
    formula_version: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(default_factory=list)


class ComparisonMetric(FinancialModel):
    metric_name: str
    company_a: ComparisonEvidence | None = None
    company_b: ComparisonEvidence | None = None
    comparison_status: Literal[
        "COMPARABLE", "INCOMPARABLE", "INSUFFICIENT_EVIDENCE"
    ]
    difference: Decimal | None = None
    higher_company_id: str | None = None
    warning_codes: list[str] = Field(default_factory=list)


class ComparisonResult(FinancialModel):
    status: Literal["OK", "OK_WITH_WARNINGS", "INSUFFICIENT_EVIDENCE"]
    question: str
    companies: list[str]
    metrics: list[ComparisonMetric] = Field(default_factory=list)
    warnings: list[ComparisonWarning] = Field(default_factory=list)

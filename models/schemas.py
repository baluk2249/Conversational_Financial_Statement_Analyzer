"""Pydantic contracts shared by retrieval and downstream agents."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FinancialModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class SourceLocator(FinancialModel):
    page: int | None = None
    label: str | None = None
    sheet: str | None = None
    cell_range: str | None = None
    paragraph: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SourceReference(FinancialModel):
    source_id: str
    source_file: str
    statement_type: str | None = None
    metric: str | None = None
    period: str | None = None
    locator: SourceLocator = Field(default_factory=SourceLocator)

    @property
    def citation(self) -> str:
        statement = (self.statement_type or "Source").replace("_", " ").title()
        parts = [statement, self.period or "Period", self.metric or "Evidence"]
        if self.locator.sheet:
            parts.append(f"Sheet: {self.locator.sheet}")
            if self.locator.cell_range:
                parts.append(f"Cell: {self.locator.cell_range}")
        elif self.locator.page is not None:
            parts.append(f"p.{self.locator.page}")
        return f"[{ ' | '.join(parts) }]"


class FinancialFact(FinancialModel):
    fact_id: str
    company_id: str
    company_name: str
    statement_type: str
    canonical_line_item: str
    metric_definition: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    fiscal_year: str
    fiscal_year_end: date | None = None
    value: Decimal
    currency: str | None = None
    unit: str = "absolute"
    source_id: str
    source_file: str
    locator: SourceLocator = Field(default_factory=SourceLocator)
    extraction_confidence: float = Field(ge=0, le=1)

    @property
    def source_reference(self) -> SourceReference:
        return SourceReference(
            source_id=self.source_id,
            source_file=self.source_file,
            statement_type=self.statement_type,
            metric=self.locator.label or self.canonical_line_item,
            period=self.fiscal_year,
            locator=self.locator,
        )


class DocumentChunk(FinancialModel):
    chunk_id: str
    company_id: str
    company_name: str
    document_type: str
    statement_type: str | None = None
    section: str
    period: str | None = None
    source_id: str
    source_file: str
    page: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def source_reference(self) -> SourceReference:
        return SourceReference(
            source_id=self.source_id,
            source_file=self.source_file,
            statement_type=self.statement_type or self.document_type,
            metric=self.section,
            period=self.period,
            locator=SourceLocator(
                page=self.page,
                sheet=self.sheet,
                cell_range=self.cell_range,
                label=self.section,
            ),
        )


class ComputedMetric(FinancialModel):
    metric_id: str
    company_id: str
    metric_name: str
    period: str
    value: Decimal | None
    display_unit: str
    formula_version: str
    input_fact_refs: list[str] = Field(default_factory=list)
    quality_status: Literal["VALID", "INSUFFICIENT_EVIDENCE", "INCOMPATIBLE_INPUTS"] = "VALID"
    currency: str | None = None
    unit: str | None = None
    warnings: list[str] = Field(default_factory=list)


class TrendResult(FinancialModel):
    trend_id: str
    company_id: str
    metric_name: str
    from_period: str
    to_period: str
    from_value: Decimal
    to_value: Decimal
    delta: Decimal
    supporting_refs: list[str] = Field(default_factory=list)


class RedFlagResult(FinancialModel):
    flag_id: str
    company_id: str
    flag_type: str
    status: str
    rule_version: str
    period: str | None = None
    supporting_refs: list[str] = Field(default_factory=list)
    explanation_code: str | None = None


class InsufficientEvidence(FinancialModel):
    status: Literal["INSUFFICIENT_EVIDENCE"] = "INSUFFICIENT_EVIDENCE"
    requested_item: str
    reason_code: str = "REQUIRED_FACTS_MISSING"
    missing_inputs: list[str] = Field(default_factory=list)
    available_evidence: list[str] = Field(default_factory=list)


class AnswerEvidence(FinancialModel):
    status: Literal["OK", "INSUFFICIENT_EVIDENCE"] = "OK"
    question: str
    company_id: str
    company_name: str | None = None
    periods: list[str] = Field(default_factory=list)
    facts: list[FinancialFact] = Field(default_factory=list)
    computed_metrics: list[ComputedMetric] = Field(default_factory=list)
    trends: list[TrendResult] = Field(default_factory=list)
    red_flags: list[RedFlagResult] = Field(default_factory=list)
    narrative_evidence: list[DocumentChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    insufficient_evidence: InsufficientEvidence | None = None
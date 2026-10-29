"""Typed application-level request, routing, and response contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from agents.comparison.schemas import ComparisonResult
from models.schemas import AnswerEvidence, FinancialModel


class QuestionIntent(StrEnum):
    EXACT_LOOKUP = "EXACT_LOOKUP"
    RATIO = "RATIO"
    TREND = "TREND"
    RED_FLAG = "RED_FLAG"
    WHY = "WHY"
    SUMMARY = "SUMMARY"
    COMPARISON = "COMPARISON"


class OrchestrationRequest(FinancialModel):
    question: str
    company_id: str
    comparison_company_id: str | None = None
    period: str | None = None
    request_id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator("question", "company_id", "comparison_company_id", "period")
    @classmethod
    def reject_blank_values(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Values must not be blank")
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def comparison_companies_must_differ(self) -> OrchestrationRequest:
        if (
            self.comparison_company_id is not None
            and self.comparison_company_id == self.company_id
        ):
            raise ValueError("Comparison companies must have different IDs")
        return self


class OrchestrationResponse(FinancialModel):
    request_id: str
    intent: QuestionIntent
    status: Literal["OK", "OK_WITH_WARNINGS", "INSUFFICIENT_EVIDENCE"]
    answer: str
    evidence: list[AnswerEvidence] = Field(default_factory=list)
    comparison: ComparisonResult | None = None
    citation_validation: Literal["VALID"]
    citation_errors: list[str] = Field(default_factory=list)

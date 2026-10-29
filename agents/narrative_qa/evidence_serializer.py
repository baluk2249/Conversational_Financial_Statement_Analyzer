"""Serialize only targeted, typed AnswerEvidence for a language model."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from agents.narrative_qa.citation_builder import CitationBuilder
from models.schemas import AnswerEvidence, ComputedMetric, TrendResult


class EvidenceSerializer:
    """Create a stable JSON evidence block with deterministic citations."""

    def __init__(self, citation_builder: CitationBuilder | None = None) -> None:
        self.citation_builder = citation_builder or CitationBuilder()

    def serialize(self, evidence: AnswerEvidence) -> str:
        payload: dict[str, Any] = {
            "status": evidence.status,
            "question": evidence.question,
            "company_id": evidence.company_id,
            "company_name": evidence.company_name,
            "periods": evidence.periods,
            "facts": [
                {
                    "fact_id": fact.fact_id,
                    "company_id": fact.company_id,
                    "company_name": fact.company_name,
                    "statement_type": fact.statement_type,
                    "line_item": fact.canonical_line_item,
                    "period": fact.fiscal_year,
                    "value": str(fact.value),
                    "currency": fact.currency,
                    "unit": fact.unit,
                    "citation": self.citation_builder.fact_citation(fact),
                }
                for fact in evidence.facts
                if fact.company_id == evidence.company_id
            ],
            "computed_metrics": [
                self._serialize_metric(metric, evidence)
                for metric in evidence.computed_metrics
                if metric.company_id == evidence.company_id
            ],
            "trends": [
                self._serialize_trend(trend, evidence)
                for trend in evidence.trends
                if trend.company_id == evidence.company_id
            ],
            "red_flags": [
                {
                    "flag_type": flag.flag_type,
                    "status": flag.status,
                    "period": flag.period,
                    "rule_version": flag.rule_version,
                    "explanation_code": flag.explanation_code,
                    "citations": self.citation_builder.flag_citations(flag, evidence),
                }
                for flag in evidence.red_flags
                if flag.company_id == evidence.company_id
            ],
            "narrative_evidence": [
                {
                    "company_id": chunk.company_id,
                    "period": chunk.period,
                    "section": chunk.section,
                    "text": chunk.text,
                    "citation": chunk.source_reference.citation,
                }
                for chunk in evidence.narrative_evidence
                if chunk.company_id == evidence.company_id
            ],
            "warnings": evidence.warnings,
        }
        if evidence.insufficient_evidence is not None:
            payload["insufficient_evidence"] = (
                evidence.insufficient_evidence.model_dump(mode="json")
            )
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _serialize_metric(
        self, metric: ComputedMetric, evidence: AnswerEvidence
    ) -> dict[str, Any]:
        return {
            "metric_id": metric.metric_id,
            "company_id": metric.company_id,
            "metric_name": metric.metric_name,
            "period": metric.period,
            "value": _decimal_string(metric.value),
            "display_unit": metric.display_unit,
            "formula_version": metric.formula_version,
            "input_fact_refs": metric.input_fact_refs,
            "quality_status": metric.quality_status,
            "warnings": metric.warnings,
            "citations": self.citation_builder.metric_citations(metric, evidence),
        }

    def _serialize_trend(
        self, trend: TrendResult, evidence: AnswerEvidence
    ) -> dict[str, Any]:
        return {
            "trend_id": trend.trend_id,
            "company_id": trend.company_id,
            "metric_name": trend.metric_name,
            "from_period": trend.from_period,
            "to_period": trend.to_period,
            "from_value": _decimal_string(trend.from_value),
            "to_value": _decimal_string(trend.to_value),
            "delta": _decimal_string(trend.delta),
            "percentage_delta": _decimal_string(trend.percentage_delta),
            "quality_status": trend.quality_status,
            "warnings": trend.warnings,
            "citations": self.citation_builder.trend_citations(trend, evidence),
        }


def _decimal_string(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None

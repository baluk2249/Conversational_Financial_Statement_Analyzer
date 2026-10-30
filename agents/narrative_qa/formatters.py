"""Deterministic safe answers for unavailable providers and missing evidence."""

from __future__ import annotations

import re
from decimal import Decimal

from agents.comparison.schemas import ComparisonEvidence, ComparisonResult
from agents.narrative_qa.citation_builder import CitationBuilder
from models.schemas import (
    AnswerEvidence,
    ComputedMetric,
    FinancialFact,
    RedFlagResult,
    TrendResult,
)

CURRENCY_SYMBOLS = {"GBP": "£", "USD": "$", "EUR": "€", "JPY": "¥"}


class SafeFallbackFormatter:
    """Render a concise answer using only facts with resolvable lineage."""

    def __init__(self, citation_builder: CitationBuilder | None = None) -> None:
        self.citations = citation_builder or CitationBuilder()

    def format(self, evidence: AnswerEvidence) -> str:
        if evidence.status == "INSUFFICIENT_EVIDENCE":
            answer = self._insufficient(evidence)
        elif _is_estimation_question(evidence.question):
            answer = (
                "The available evidence does not support a forecast or estimate; "
                "I will not extrapolate a financial value."
            )
        elif is_causal_question(evidence.question) and not any(
            chunk.company_id == evidence.company_id
            for chunk in evidence.narrative_evidence
        ):
            answer = (
                "The available statements do not provide evidence explaining the "
                "cause; I will not infer one."
            )
        elif is_causal_question(evidence.question):
            narrative = [
                chunk for chunk in evidence.narrative_evidence
                if chunk.company_id == evidence.company_id
            ]
            answer = " ".join(
                f'The retrieved narrative states: "{chunk.text}" '
                f"{self.citations.narrative_citation(chunk)}"
                for chunk in narrative
            )
        elif _is_comparison_question(evidence.question):
            answer = (
                "A comparison cannot be established because this evidence bundle "
                "does not include verified, company-scoped comparison results."
            )
        else:
            statements = [
                self._format_fact(fact)
                for fact in evidence.facts
                if fact.company_id == evidence.company_id
            ]
            statements.extend(
                self._format_metric(metric, evidence)
                for metric in evidence.computed_metrics
                if metric.company_id == evidence.company_id
            )
            statements.extend(
                self._format_trend(trend, evidence)
                for trend in evidence.trends
                if trend.company_id == evidence.company_id
            )
            statements.extend(
                self._format_flag(flag, evidence)
                for flag in evidence.red_flags
                if flag.company_id == evidence.company_id
            )
            statements = [statement for statement in statements if statement]
            answer = " ".join(statements) or (
                "The available evidence does not establish the requested "
                "financial information."
            )
        return append_warnings(answer, evidence)

    def format_comparison(
        self,
        result: ComparisonResult,
        *,
        upstream_insufficient: bool = False,
    ) -> str:
        if upstream_insufficient or result.status == "INSUFFICIENT_EVIDENCE":
            answer = (
                "Required company evidence is missing. The comparison cannot "
                "be established; I will not estimate."
            )
        else:
            statements: list[str] = []
            for metric in result.metrics:
                if metric.comparison_status == "COMPARABLE":
                    left = self._format_comparison_value(metric.company_a)
                    right = self._format_comparison_value(metric.company_b)
                    if left is not None and right is not None:
                        statements.append(
                            f"{_label(metric.metric_name)}: {left}; versus {right}."
                        )
                elif metric.comparison_status == "INCOMPARABLE":
                    statements.append(
                        f"{_label(metric.metric_name)} values are not comparable."
                    )
                else:
                    statements.append(
                        f"{_label(metric.metric_name)} cannot be compared; "
                        "required evidence is missing."
                    )
            answer = " ".join(statements) or (
                "The available evidence does not establish a comparison."
            )

        warning_codes = list(dict.fromkeys(
            warning.code for warning in result.warnings
        ))
        if warning_codes:
            answer = f"{answer.rstrip()} Comparison warnings: {', '.join(warning_codes)}."
        return answer

    @staticmethod
    def _format_comparison_value(
        evidence: ComparisonEvidence | None,
    ) -> str | None:
        if evidence is None or not evidence.source_references:
            return None
        value = evidence.value
        display_unit = evidence.display_unit or evidence.unit
        if display_unit == "percent":
            formatted_value = f"{value * Decimal('100')}%"
        else:
            formatted_value = f"{value}"
        context = " ".join(
            value for value in (evidence.currency, evidence.unit) if value
        )
        citations = " ".join(
            source.citation for source in evidence.source_references
        )
        label = evidence.company_name or evidence.company_id
        return " ".join(
            part for part in (label, formatted_value, context, citations) if part
        )

    def _insufficient(self, evidence: AnswerEvidence) -> str:
        period = ""
        requested = ""
        if evidence.insufficient_evidence is not None:
            requested = evidence.insufficient_evidence.requested_item
            if requested:
                requested = f" {requested}"
        if evidence.periods:
            period = f" for {', '.join(evidence.periods)}"
        elif evidence.insufficient_evidence is not None:
            match = re.search(r"\bFY\d{4}\b", evidence.question, re.IGNORECASE)
            if match:
                period = f" for {match.group(0)}"
        return (
            f"The requested{requested} information{period} cannot be determined "
            "from the available statements; I will not estimate it."
        )

    def _format_fact(self, fact: FinancialFact) -> str:
        citation = self.citations.fact_citation(fact)
        amount = _format_fact_value(fact)
        label = _label(fact.canonical_line_item)
        return (
            f"Reported {label} for {fact.fiscal_year} was {amount} "
            f"(unit: {fact.unit}) {citation}."
        )

    def _format_metric(self, metric: ComputedMetric, evidence: AnswerEvidence) -> str:
        if (
            metric.quality_status != "VALID"
            or metric.value is None
        ):
            return ""
        citations = self.citations.metric_citations(metric, evidence)
        if not citations:
            return ""
        value = _format_metric_value(metric.value, metric.display_unit)
        citation_text = " ".join(citations)
        inputs = ", ".join(metric.input_fact_refs)
        return (
            f"Calculated {_label(metric.metric_name)} for {metric.period} was "
            f"{value} using {metric.formula_version}; inputs: {inputs}. "
            f"{citation_text}"
        )

    def _format_trend(self, trend: TrendResult, evidence: AnswerEvidence) -> str:
        if (
            trend.quality_status != "VALID"
            or trend.from_value is None
            or trend.to_value is None
        ):
            return ""
        citations = self.citations.trend_citations(trend, evidence)
        if not citations:
            return ""
        display_unit = _trend_display_unit(trend, evidence)
        before = _format_metric_value(trend.from_value, display_unit)
        after = _format_metric_value(trend.to_value, display_unit)
        citation_text = " ".join(citations)
        return (
            f"{_label(trend.metric_name)} changed from {before} in "
            f"{trend.from_period} to {after} in {trend.to_period}. "
            f"{citation_text}"
        )

    def _format_flag(self, flag: RedFlagResult, evidence: AnswerEvidence) -> str:
        citations = self.citations.flag_citations(flag, evidence)
        citation_text = f" {' '.join(citations)}" if citations else ""
        if flag.status == "TRIGGERED":
            return (
                f"The configured { _label(flag.flag_type).lower() } rule "
                f"identified a warning for {flag.period or 'the available period'} "
                f"(rule: {flag.rule_version}); this is an indicator for review, "
                f"not a conclusion.{citation_text}"
            )
        if flag.status == "NOT_TRIGGERED":
            return (
                f"The configured {_label(flag.flag_type).lower()} rule was not "
                f"triggered for {flag.period or 'the available period'} "
                f"(rule: {flag.rule_version}).{citation_text}"
            )
        if flag.status == "INSUFFICIENT_EVIDENCE":
            return (
                f"The {_label(flag.flag_type).lower()} check is inconclusive "
                f"; required evidence is unavailable "
                f"(rule: {flag.rule_version}).{citation_text}"
            )
        return ""


def append_warnings(answer: str, evidence: AnswerEvidence) -> str:
    if not evidence.warnings:
        return answer
    missing = [
        warning
        for warning in evidence.warnings
        if warning.casefold() not in answer.casefold()
    ]
    if not missing:
        return answer
    return (
        f"{answer.rstrip()} Upstream warnings affecting interpretation: "
        f"{'; '.join(missing)}."
    )


def _format_fact_value(fact: FinancialFact) -> str:
    number = _format_decimal(fact.value)
    symbol = CURRENCY_SYMBOLS.get(fact.currency or "")
    currency = fact.currency or "currency not specified"
    if symbol:
        return f"{symbol}{number}"
    return f"{number} {currency}"


def _format_metric_value(value: Decimal, display_unit: str) -> str:
    if display_unit == "percent":
        return f"{_format_decimal(value * Decimal('100'))}%"
    if display_unit == "days":
        return f"{_format_decimal(value)} days"
    if display_unit == "ratio":
        return f"{_format_decimal(value)}"
    return f"{_format_decimal(value)} {display_unit}"


def _format_decimal(value: Decimal) -> str:
    return f"{value:,}"


def _trend_display_unit(trend: TrendResult, evidence: AnswerEvidence) -> str:
    for metric in evidence.computed_metrics:
        if (
            metric.company_id == trend.company_id
            and metric.metric_name == trend.metric_name
            and metric.period in {trend.from_period, trend.to_period}
        ):
            return metric.display_unit
    return "ratio"


def _label(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _is_comparison_question(question: str) -> bool:
    return bool(
        re.search(r"\b(compare|comparison|versus|vs\.?)\b", question, re.IGNORECASE)
    )


def _is_estimation_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(estimate|forecast|project|predict|extrapolat\w*|next year|"
            r"future|expected revenue|expected earnings)\b",
            question,
            re.IGNORECASE,
        )
    )


def requires_deterministic_refusal(evidence: AnswerEvidence) -> bool:
    """Avoid model generation when the evidence contract cannot support the ask."""
    return _is_comparison_question(evidence.question) or _is_estimation_question(
        evidence.question
    )


def is_causal_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:why|cause\w*|reason\w*|what led to|what drove|"
            r"what factors?|what explains?)\b",
            question,
            re.IGNORECASE,
        )
    )

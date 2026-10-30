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
        trend_request = (
            _is_trend_question(evidence.question)
            and not _is_comparison_question(evidence.question)
            and not is_causal_question(evidence.question)
        )
        if evidence.status == "INSUFFICIENT_EVIDENCE" and trend_request:
            answer = self._format_trend_answer(evidence)
        elif evidence.status == "INSUFFICIENT_EVIDENCE":
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
        elif trend_request:
            answer = self._format_trend_answer(evidence)
        else:
            statements = self._format_facts(
                [
                    fact for fact in evidence.facts
                    if fact.company_id == evidence.company_id
                ]
            )
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
            statements = list(dict.fromkeys(statements))
            answer = " ".join(statements) or (
                "The available evidence does not establish the requested "
                "financial information."
            )
        return append_warnings(answer, evidence)

    def _format_facts(self, facts: list[FinancialFact]) -> list[str]:
        if not facts:
            return []
        if len(facts) == 1:
            return [self._format_fact(facts[0])]

        grouped_measurements: dict[tuple[object, ...], list[FinancialFact]] = {}
        for fact in facts:
            key = (
                fact.fiscal_year,
                fact.canonical_line_item,
                fact.value,
                fact.currency,
                fact.unit,
                fact.period_start,
                fact.period_end,
                fact.fiscal_year_end,
                fact.metric_definition,
                fact.statement_type,
            )
            grouped_measurements.setdefault(key, []).append(fact)

        statement_order = {
            "income_statement": 0,
            "balance_sheet": 1,
            "cash_flow_statement": 2,
        }
        statements_by_group: dict[tuple[str, str], list[str]] = {}
        for measurement_facts in grouped_measurements.values():
            primary_fact = min(
                measurement_facts,
                key=lambda fact: statement_order.get(fact.statement_type, 3),
            )
            citations = list(dict.fromkeys(
                self.citations.fact_citation(fact)
                for fact in measurement_facts
            ))
            label = _humanize_label(primary_fact.canonical_line_item)
            amount = _format_fact_amount(primary_fact)
            statements_by_group.setdefault(
                (primary_fact.fiscal_year, primary_fact.statement_type), []
            ).append(
                f"Reported {label} was {amount} {' '.join(citations)}"
            )

        ordered_groups = sorted(
            statements_by_group.items(),
            key=lambda item: (
                item[0][0],
                statement_order.get(item[0][1], 3),
                item[0][1],
            ),
        )
        formatted = ["Reported figures, grouped by period and statement:"]
        for (period, statement_type), entries in ordered_groups:
            statement_name = _statement_label(statement_type)
            formatted.append(
                f"- {period} {statement_name}: {'; '.join(entries)}."
            )
        return formatted

    def _format_trend_answer(self, evidence: AnswerEvidence) -> str:
        trends_by_metric: dict[str, list[TrendResult]] = {}
        for trend in evidence.trends:
            if (
                trend.company_id == evidence.company_id
                and trend.quality_status == "VALID"
                and trend.from_value is not None
                and trend.to_value is not None
            ):
                trends_by_metric.setdefault(trend.metric_name, []).append(trend)

        grouped_metrics: dict[str, list[str]] = {}
        metric_groups = {
            "current_ratio": "Liquidity",
            "quick_ratio": "Liquidity",
            "operating_cash_flow_ratio": "Liquidity",
            "debt_to_equity": "Leverage",
            "debt_to_assets": "Leverage",
            "interest_coverage": "Leverage",
            "gross_margin": "Profitability",
            "operating_margin": "Profitability",
            "net_margin": "Profitability",
            "roa": "Profitability",
            "roe": "Profitability",
            "dso": "Efficiency",
            "inventory_days": "Efficiency",
            "asset_turnover": "Efficiency",
        }
        for metric_name, trends in trends_by_metric.items():
            ordered_trends = sorted(
                trends,
                key=lambda trend: (
                    _period_sort_key(trend.from_period),
                    _period_sort_key(trend.to_period),
                ),
            )
            points: list[tuple[str, Decimal]] = []
            if ordered_trends:
                points.append(
                    (ordered_trends[0].from_period, ordered_trends[0].from_value)
                )
                points.extend(
                    (trend.to_period, trend.to_value)
                    for trend in ordered_trends
                )
            if len(points) < 2:
                continue

            display_unit = _trend_display_unit(ordered_trends[0], evidence)
            metrics_by_id = {
                metric.metric_id: metric
                for metric in evidence.computed_metrics
                if metric.company_id == evidence.company_id
            }
            citations_by_period: dict[str, list[str]] = {}
            for trend in ordered_trends:
                for reference in trend.supporting_refs:
                    metric = metrics_by_id.get(reference)
                    if metric is None:
                        continue
                    citations_by_period.setdefault(metric.period, []).extend(
                        self.citations.metric_citations(metric, evidence)
                    )
            formatted_points: list[str] = []
            for period, value in points:
                citations = list(dict.fromkeys(citations_by_period.get(period, [])))
                if not citations:
                    formatted_points = []
                    break
                formatted_points.append(
                    f"  - {period}: {_format_metric_value(value, display_unit)} "
                    f"{' '.join(citations)}"
                )
            if not formatted_points:
                continue
            directions = [
                "increased" if trend.to_value > trend.from_value else (
                    "decreased" if trend.to_value < trend.from_value else "was unchanged"
                )
                for trend in ordered_trends
            ]
            if len(set(directions)) == 1:
                movement = {
                    "increased": "increased in each period",
                    "decreased": "decreased in each period",
                    "was unchanged": "was unchanged in each period",
                }[directions[0]]
            else:
                movement = "changed over the reporting periods"
            statement = (
                f"- {_humanize_label(metric_name)} {movement}:\n"
                + "\n".join(formatted_points)
            )
            group = metric_groups.get(metric_name, "Other measures")
            grouped_metrics.setdefault(group, []).append(statement)

        if grouped_metrics:
            sections = ["Financial trends by area:"]
            group_order = (
                "Liquidity",
                "Profitability",
                "Leverage",
                "Efficiency",
                "Other measures",
            )
            for group in group_order:
                statements = grouped_metrics.get(group)
                if statements:
                    sections.append(
                        f"- {group}:\n"
                        + "\n".join(
                            "\n".join(
                                f"  {line}" for line in statement.splitlines()
                            )
                            for statement in statements
                        )
                    )
            return "\n".join(sections)

        if not evidence.trends:
            valid_trends = []
        else:
            valid_trends = [
                self._format_trend(trend, evidence)
                for trend in evidence.trends
                if trend.company_id == evidence.company_id
                and trend.quality_status == "VALID"
            ]
            valid_trends = [statement for statement in valid_trends if statement]
        if valid_trends:
            return "Supported period-over-period changes: " + " ".join(
                valid_trends
            )
        if len(set(evidence.periods)) < 2:
            period = evidence.periods[0] if evidence.periods else "the available data"
            return (
                "I couldn't identify financial trends because the available "
                f"evidence covers only {period}. Trends require comparable "
                "figures from at least two reporting periods."
            )
        return (
            "I couldn't establish financial trends because comparable "
            "figures or required inputs were missing or incompatible "
            "across the available periods."
        )

    def format_summary(self, evidence: AnswerEvidence, section: str) -> str:
        """Create a concise evidence-only dashboard summary without figures."""
        if section.casefold().replace("-", " ") == "red flags":
            triggered = [
                _label(flag.flag_type).lower()
                for flag in evidence.red_flags
                if flag.status == "TRIGGERED"
            ]
            not_triggered = [
                _label(flag.flag_type).lower()
                for flag in evidence.red_flags
                if flag.status == "NOT_TRIGGERED"
            ]
            inconclusive = [
                _label(flag.flag_type).lower()
                for flag in evidence.red_flags
                if flag.status == "INSUFFICIENT_EVIDENCE"
            ]
            statements = []
            if triggered:
                statements.append(
                    "Configured indicators requiring review include "
                    f"{_join_labels(triggered)}."
                )
            if not_triggered:
                statements.append(
                    "The configured checks not triggered are "
                    f"{_join_labels(not_triggered)}."
                )
            if inconclusive:
                statements.append(
                    "The following checks are inconclusive because required "
                    f"evidence is missing: {_join_labels(inconclusive)}."
                )
            if not statements:
                statements.append(
                    "No indicator results are available, so the evidence "
                    "cannot establish whether configured checks triggered."
                )
            if evidence.red_flags:
                statements.append(
                    "A triggered indicator is a prompt for further review, "
                    "not a conclusion."
                )
            return append_warnings(" ".join(statements), evidence)

        available = [
            _label(metric.metric_name).lower()
            for metric in evidence.computed_metrics
            if metric.company_id == evidence.company_id
            and metric.quality_status == "VALID"
            and metric.value is not None
        ]
        unavailable = [
            metric
            for metric in evidence.computed_metrics
            if metric.company_id == evidence.company_id
            and (metric.quality_status != "VALID" or metric.value is None)
        ]
        statements = []
        section_name = section.lower()
        if available:
            statements.append(
                f"Available {section_name} measures include "
                f"{_join_labels(available)}."
            )
        else:
            statements.append(
                f"No {section_name} measures could be calculated from the "
                "available evidence."
            )
        if unavailable:
            missing = []
            for metric in unavailable:
                reasons = [
                    warning.removeprefix("MISSING_INPUT:").replace("_", " ")
                    for warning in metric.warnings
                    if warning.startswith("MISSING_INPUT:")
                ]
                label = _label(metric.metric_name).lower()
                if reasons:
                    label += f" (missing {', '.join(reasons)})"
                missing.append(label)
            statements.append(
                "The following measures are unavailable: "
                f"{_join_labels(missing)}."
            )
        if evidence.trends:
            directions = []
            for trend in evidence.trends:
                if (
                    trend.company_id != evidence.company_id
                    or trend.quality_status != "VALID"
                    or trend.delta is None
                ):
                    continue
                direction = "increased" if trend.delta > 0 else (
                    "decreased" if trend.delta < 0 else "was unchanged"
                )
                directions.append(
                    f"{_label(trend.metric_name).lower()} {direction}"
                )
            if directions:
                statements.append(
                    "The available period-over-period results show that "
                    f"{_join_labels(directions)}."
                )
            else:
                statements.append(
                    "Period-over-period direction cannot be established from "
                    "the available trend results."
                )
        else:
            statements.append(
                "No period-over-period trend evidence is included, so the "
                "direction of change cannot be assessed."
            )
        return append_warnings(" ".join(statements), evidence)

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
    def format_comparison_summary(result: ComparisonResult) -> str:
        """Summarize comparison coverage without restating numeric values."""
        if result.status == "INSUFFICIENT_EVIDENCE":
            return (
                "Required company evidence is missing, so no comparison can "
                "be established."
            )

        comparable = [
            _label(metric.metric_name).lower()
            for metric in result.metrics
            if metric.comparison_status == "COMPARABLE"
        ]
        incomparable = [
            _label(metric.metric_name).lower()
            for metric in result.metrics
            if metric.comparison_status == "INCOMPARABLE"
        ]
        missing = [
            _label(metric.metric_name).lower()
            for metric in result.metrics
            if metric.comparison_status == "INSUFFICIENT_EVIDENCE"
        ]
        statements = []
        if comparable:
            statements.append(
                "Comparable measures include "
                f"{_join_labels(comparable)}."
            )
        if incomparable:
            statements.append(
                "The following measures are not comparable: "
                f"{_join_labels(incomparable)}."
            )
        if missing:
            statements.append(
                "A comparison could not be completed for "
                f"{_join_labels(missing)} because evidence is missing."
            )
        if not statements:
            statements.append("No comparable results are available.")
        warning_codes = list(dict.fromkeys(
            warning.code for warning in result.warnings
        ))
        if warning_codes:
            statements.append(
                "Comparison warnings: " + ", ".join(warning_codes) + "."
            )
        return " ".join(statements)

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
        amount = _format_fact_amount(fact)
        label = _label(fact.canonical_line_item)
        return (
            f"Reported {label} for {fact.fiscal_year} was {amount} "
            f"{citation}."
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
    number = _format_decimal(abs(fact.value))
    symbol = CURRENCY_SYMBOLS.get(fact.currency or "")
    currency = fact.currency or "currency not specified"
    sign = "-" if fact.value < 0 else ""
    if symbol:
        return f"{sign}{symbol}{number}"
    return f"{sign}{number} {currency}"


def _format_fact_amount(fact: FinancialFact) -> str:
    amount = _format_fact_value(fact)
    unit = fact.unit.strip().casefold()
    unit_names = {
        "million": "million",
        "millions": "million",
        "thousand": "thousand",
        "thousands": "thousand",
        "billion": "billion",
        "billions": "billion",
    }
    if unit in unit_names:
        return f"{amount} {unit_names[unit]}"
    if unit not in {"", "absolute", "unit", "units", "currency"}:
        return f"{amount} ({fact.unit})"
    return amount


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


def _humanize_label(value: str) -> str:
    label = _label(value)
    for word in ("Of", "And", "From", "For", "In"):
        label = label.replace(f" {word} ", f" {word.lower()} ")
    return label


def _statement_label(statement_type: str) -> str:
    labels = {
        "income_statement": "income statement",
        "balance_sheet": "balance sheet",
        "cash_flow_statement": "cash flow statement",
    }
    return labels.get(statement_type, statement_type.replace("_", " "))


def _period_sort_key(period: str) -> tuple[int, str]:
    match = re.search(r"\d{4}", period)
    return (int(match.group(0)) if match else -1, period)


def _join_labels(labels: list[str]) -> str:
    if len(labels) < 2:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _is_comparison_question(question: str) -> bool:
    return bool(
        re.search(r"\b(compare|comparison|versus|vs\.?)\b", question, re.IGNORECASE)
    )


def _is_trend_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(trend\w*|declin\w*|increas\w*|decreas\w*|change\w*|"
            r"over time|period.over.period)\b",
            question,
            re.IGNORECASE,
        )
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

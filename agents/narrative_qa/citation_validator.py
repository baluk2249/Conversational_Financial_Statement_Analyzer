"""Reject numeric or causal statements that cannot be traced to evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from agents.narrative_qa.citation_builder import CitationBuilder
from agents.comparison.schemas import ComparisonResult
from models.schemas import AnswerEvidence, ComputedMetric, TrendResult

CITATION_PATTERN = re.compile(r"\[[^\[\]]+\]")
NUMBER_PATTERN = re.compile(
    r"(?<!\w)(?:[£$€¥]\s*)?-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
    r"(?:\s*(?:%|x|[kKmMbB]|thousands?|millions?|billions?|trillions?))?"
    r"(?!\w)",
)
CAUSAL_PATTERN = re.compile(
    r"\b(?:because|caused by|due to|driven by|resulted from|"
    r"as a result of|attributable to|led to|contributed to|owing to)\b",
    re.IGNORECASE,
)
CAUSAL_RELATION_PATTERN = re.compile(
    r"\b(?:because|cause[sd]?|due to|drive[sn]?|result(?:ed|ing)?|"
    r"reduc(?:e|ed|ing)|increas(?:e|ed|ing)|contribut(?:e|ed|ing)|"
    r"affect(?:ed|s|ing)?|impact(?:ed|s|ing)?|lead|led)\b",
    re.IGNORECASE,
)
STOP_WORDS = {
    "about", "after", "also", "and", "are", "because", "been", "being",
    "company", "could", "due", "from", "have", "into", "its", "more",
    "period", "result", "than", "that", "the", "their", "there", "these",
    "this", "through", "was", "were", "which", "with",
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class CitationValidator:
    """Validate citations, numbers, and causal claims in a proposed answer."""

    def __init__(self, citation_builder: CitationBuilder | None = None) -> None:
        self.citations = citation_builder or CitationBuilder()

    def validate(
        self, answer: str, evidence: AnswerEvidence
    ) -> ValidationResult:
        errors: list[str] = []
        if not answer.strip():
            return ValidationResult(False, ["The answer is empty"])

        citation_values = self._citation_values(evidence)
        citations_in_answer: list[tuple[str, int, int]] = []
        for match in CITATION_PATTERN.finditer(answer):
            citation = match.group(0)
            if citation not in citation_values:
                if "|" in citation:
                    errors.append(f"Unresolvable citation: {citation}")
                continue
            citations_in_answer.append((citation, match.start(), match.end()))

        scrubbed = CITATION_PATTERN.sub("", answer)
        for match in NUMBER_PATTERN.finditer(scrubbed):
            value = _parse_number(match.group(0))
            if value is None:
                errors.append(f"Unrecognized numeric claim: {match.group(0)}")
                continue
            original_start = _scrubbed_to_original_index(answer, match.start())
            nearby_start = max(0, original_start - 180)
            nearby_end = min(len(answer), original_start + 180)
            nearby_citations = [
                citation
                for citation, start, end in citations_in_answer
                if start < nearby_end and end > nearby_start
            ]
            matching_citations = [
                citation
                for citation in nearby_citations
                if value in citation_values[citation]
            ]
            if evidence.status != "OK" or not matching_citations:
                errors.append(
                    f"Numeric claim {match.group(0)!r} is not grounded by a "
                    "nearby, supporting citation"
                )

        errors.extend(self._causal_claim_errors(answer, evidence))
        return ValidationResult(not errors, errors)

    def validate_comparison(
        self, answer: str, result: ComparisonResult
    ) -> ValidationResult:
        """Validate comparison claims against each metric's source lineage."""
        if not answer.strip():
            return ValidationResult(False, ["The answer is empty"])

        citation_values: dict[str, set[Decimal]] = {}
        if result.status != "INSUFFICIENT_EVIDENCE":
            for metric in result.metrics:
                for evidence in (metric.company_a, metric.company_b):
                    if evidence is None:
                        continue
                    supported_values = {evidence.value}
                    if evidence.normalized_value is not None:
                        supported_values.add(evidence.normalized_value)
                    if evidence.display_unit == "percent":
                        supported_values.add(evidence.value * Decimal("100"))
                    for source in evidence.source_references:
                        citation_values.setdefault(source.citation, set()).update(
                            supported_values
                        )

        errors: list[str] = []
        citations_in_answer: list[tuple[str, int, int]] = []
        for match in CITATION_PATTERN.finditer(answer):
            citation = match.group(0)
            if citation not in citation_values:
                if "|" in citation:
                    errors.append(f"Unresolvable citation: {citation}")
                continue
            citations_in_answer.append((citation, match.start(), match.end()))

        scrubbed = CITATION_PATTERN.sub("", answer)
        for match in NUMBER_PATTERN.finditer(scrubbed):
            value = _parse_number(match.group(0))
            if value is None:
                errors.append(f"Unrecognized numeric claim: {match.group(0)}")
                continue
            original_start = _scrubbed_to_original_index(answer, match.start())
            nearby_start = max(0, original_start - 180)
            nearby_end = min(len(answer), original_start + 180)
            nearby_citations = [
                citation
                for citation, start, end in citations_in_answer
                if start < nearby_end and end > nearby_start
            ]
            if result.status == "INSUFFICIENT_EVIDENCE" or not any(
                value in citation_values[citation] for citation in nearby_citations
            ):
                errors.append(
                    f"Numeric claim {match.group(0)!r} is not grounded by a "
                    "nearby, supporting comparison citation"
                )
        if CAUSAL_PATTERN.search(answer):
            errors.append(
                "Causal claims are not supported by comparison evidence"
            )
        return ValidationResult(not errors, errors)

    def _citation_values(
        self, evidence: AnswerEvidence
    ) -> dict[str, set[Decimal]]:
        values: dict[str, set[Decimal]] = {}
        if evidence.status != "OK":
            return values

        for fact in evidence.facts:
            if fact.company_id != evidence.company_id:
                continue
            citation = self.citations.fact_citation(fact)
            values.setdefault(citation, set()).add(fact.value)

        for metric in evidence.computed_metrics:
            if (
                metric.company_id != evidence.company_id
                or metric.quality_status != "VALID"
                or metric.value is None
            ):
                continue
            self._add_values(
                values,
                self.citations.metric_citations(metric, evidence),
                _metric_values(metric),
            )

        for trend in evidence.trends:
            if (
                trend.company_id != evidence.company_id
                or trend.quality_status != "VALID"
            ):
                continue
            self._add_values(
                values,
                self.citations.trend_citations(trend, evidence),
                _trend_values(trend, evidence.computed_metrics),
            )

        for chunk in evidence.narrative_evidence:
            if chunk.company_id != evidence.company_id:
                continue
            citation = chunk.source_reference.citation
            values.setdefault(citation, set())
            for match in NUMBER_PATTERN.finditer(chunk.text):
                number = _parse_number(match.group(0))
                if number is not None:
                    values.setdefault(citation, set()).add(number)

        return values

    @staticmethod
    def _add_values(
        mapping: dict[str, set[Decimal]],
        citations: list[str],
        supported_values: set[Decimal],
    ) -> None:
        for citation in citations:
            mapping.setdefault(citation, set()).update(supported_values)

    @staticmethod
    def _causal_claim_errors(
        answer: str, evidence: AnswerEvidence
    ) -> list[str]:
        errors: list[str] = []
        narrative = [
            chunk.text.casefold()
            for chunk in evidence.narrative_evidence
            if chunk.company_id == evidence.company_id
        ]
        for match in CAUSAL_PATTERN.finditer(answer):
            clause_end = re.search(r"[.!?]", answer[match.end():])
            end = match.end() + clause_end.start() if clause_end else len(answer)
            prefix_start = max(0, match.start() - 180)
            prefix = answer[prefix_start:match.start()]
            if not re.search(
                r"\b(?:margin|revenue|profit|earnings|cash flow|"
                r"liabilit\w*|assets?|debt|ratio|liquidit\w*|"
                r"expenses?|costs?|sales|income|inventory|receivables?)\b",
                prefix,
                re.IGNORECASE,
            ):
                continue
            cause_clause = CITATION_PATTERN.sub(
                "", answer[match.start():end]
            ).casefold()
            if not any(_causal_clause_supported(cause_clause, text) for text in narrative):
                errors.append(
                    "Causal claim is not explicitly supported by retrieved "
                    "narrative evidence"
                )
        return errors


def _metric_values(metric: ComputedMetric) -> set[Decimal]:
    if metric.value is None:
        return set()
    values = {metric.value}
    if metric.display_unit == "percent":
        values.add(metric.value * Decimal("100"))
    return values


def _trend_values(
    trend: TrendResult, metrics: list[ComputedMetric]
) -> set[Decimal]:
    result = {
        value
        for value in (
            trend.from_value,
            trend.to_value,
            trend.delta,
            trend.absolute_delta,
            trend.percentage_delta,
        )
        if value is not None
    }
    if any(
        metric.metric_name == trend.metric_name
        and metric.period in {trend.from_period, trend.to_period}
        and metric.display_unit == "percent"
        for metric in metrics
    ):
        result.update(
            value * Decimal("100")
            for value in (trend.from_value, trend.to_value, trend.delta)
            if value is not None
        )
    return result


def _parse_number(value: str) -> Decimal | None:
    unit = re.search(
        r"\s*(thousands?|millions?|billions?|trillions?|[kmb])\s*$",
        value,
        re.IGNORECASE,
    )
    multiplier = Decimal("1")
    if unit is not None:
        unit_name = unit.group(1).casefold()
        multiplier = {
            "k": Decimal("1000"),
            "thousand": Decimal("1000"),
            "thousands": Decimal("1000"),
            "m": Decimal("1000000"),
            "million": Decimal("1000000"),
            "millions": Decimal("1000000"),
            "b": Decimal("1000000000"),
            "billion": Decimal("1000000000"),
            "billions": Decimal("1000000000"),
            "trillion": Decimal("1000000000000"),
            "trillions": Decimal("1000000000000"),
        }[unit_name]
        value = value[:unit.start()]
    cleaned = re.sub(r"[£$€¥,%xX\s]", "", value).replace(",", "")
    try:
        return Decimal(cleaned) * multiplier
    except InvalidOperation:
        return None


def _scrubbed_to_original_index(original: str, scrubbed_index: int) -> int:
    cursor = 0
    original_index = 0
    while original_index < len(original) and cursor < scrubbed_index:
        if original[original_index] == "[":
            citation_end = original.find("]", original_index)
            if citation_end != -1:
                original_index = citation_end + 1
                continue
        cursor += 1
        original_index += 1
    return original_index


def _causal_clause_supported(clause: str, narrative: str) -> bool:
    if not CAUSAL_RELATION_PATTERN.search(narrative):
        return False
    cause = re.split(
        r"\b(?:because|caused by|due to|driven by|resulted from|"
        r"as a result of|attributable to|led to|contributed to|owing to)\b",
        clause,
        maxsplit=1,
    )[-1]
    cause_words = {
        word for word in re.findall(r"[a-z]{4,}", cause)
        if word not in STOP_WORDS
    }
    narrative_words = set(re.findall(r"[a-z]{4,}", narrative))
    if not cause_words:
        return False
    return len(cause_words & narrative_words) / len(cause_words) >= 0.6

"""Comparison facade consuming two isolated, grounded AnswerEvidence bundles."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

from agents.comparison.comparator import compare_values
from agents.comparison.schemas import (
    ComparisonEvidence,
    ComparisonMetric,
    ComparisonRequest,
    ComparisonResult,
    ComparisonWarning,
)
from agents.comparison.validators import NON_MONETARY_UNITS, validate_comparability
from models.schemas import AnswerEvidence, ComputedMetric, FinancialFact, SourceReference


class ComparisonAgent:
    """Validate comparability, then compare metrics using Decimal arithmetic."""

    def compare(
        self,
        evidence_a: AnswerEvidence,
        evidence_b: AnswerEvidence,
        request: ComparisonRequest | None = None,
        *,
        metric_names: Iterable[str] | None = None,
        period: str | None = None,
    ) -> ComparisonResult:
        """Compare requested metrics while keeping each company's evidence isolated."""
        if request is not None and (metric_names is not None or period is not None):
            raise ValueError("Pass either a ComparisonRequest or metric_names/period, not both")
        if request is None:
            request = ComparisonRequest(
                question="Compare the supplied financial evidence.",
                metric_names=list(metric_names or ()),
                period=period,
            )
        trend_mode = request.compare_trends or _question_requests_trend(request.question)

        company_a_id, company_b_id = evidence_a.company_id, evidence_b.company_id
        warnings = self._upstream_warnings(evidence_a, evidence_b)
        if company_a_id == company_b_id:
            warnings.append(
                ComparisonWarning(
                    code="COMPANY_IDENTITY_MISMATCH",
                    message="The two evidence bundles must identify different companies.",
                    company_ids=[company_a_id, company_b_id],
                )
            )

        names = self._metric_names(evidence_a, evidence_b, request.metric_names)
        metrics: list[ComparisonMetric] = []
        for name in names:
            candidate_a = self._select(evidence_a, name, request.period, trend_mode)
            candidate_b = self._select(evidence_b, name, request.period, trend_mode)
            if isinstance(candidate_a, str) or isinstance(candidate_b, str):
                reason_a = candidate_a if isinstance(candidate_a, str) else None
                reason_b = candidate_b if isinstance(candidate_b, str) else None
                warning = ComparisonWarning(
                    code="MISSING_COMPARISON_EVIDENCE",
                    message=(
                        f"Evidence for {name} is missing or ambiguous"
                        f"{': ' + '; '.join(item for item in (reason_a, reason_b) if item) if reason_a or reason_b else ''}."
                    ),
                    metric_name=name,
                    company_ids=[
                        company_id
                        for company_id, missing in (
                            (company_a_id, candidate_a is None or isinstance(candidate_a, str)),
                            (company_b_id, candidate_b is None or isinstance(candidate_b, str)),
                        )
                        if missing
                    ],
                )
                warnings.append(warning)
                metrics.append(
                    ComparisonMetric(
                        metric_name=name,
                        company_a=candidate_a if isinstance(candidate_a, ComparisonEvidence) else None,
                        company_b=candidate_b if isinstance(candidate_b, ComparisonEvidence) else None,
                        comparison_status="INSUFFICIENT_EVIDENCE",
                        warning_codes=["MISSING_COMPARISON_EVIDENCE"],
                    )
                )
                continue

            if candidate_a is None or candidate_b is None:
                warning = ComparisonWarning(
                    code="MISSING_COMPARISON_EVIDENCE",
                    message=f"Evidence for {name} is missing from one or both companies.",
                    metric_name=name,
                    company_ids=[
                        company_id
                        for company_id, candidate in (
                            (company_a_id, candidate_a),
                            (company_b_id, candidate_b),
                        )
                        if candidate is None
                    ],
                )
                warnings.append(warning)
                metrics.append(
                    ComparisonMetric(
                        metric_name=name,
                        company_a=candidate_a,
                        company_b=candidate_b,
                        comparison_status="INSUFFICIENT_EVIDENCE",
                        warning_codes=["MISSING_COMPARISON_EVIDENCE"],
                    )
                )
                continue

            metadata_a = self._metadata(evidence_a, name, request.period, trend_mode)
            metadata_b = self._metadata(evidence_b, name, request.period, trend_mode)
            compatibility = validate_comparability(
                candidate_a,
                candidate_b,
                metric_name=name,
                company_a_id=company_a_id,
                company_b_id=company_b_id,
                non_monetary=self._is_non_monetary(candidate_a, candidate_b),
                same_period_range=self._period_range_match(metadata_a, metadata_b),
                fiscal_year_end_match=self._fiscal_year_end_match(metadata_a, metadata_b),
            )
            warnings.extend(compatibility.warnings)
            metrics.append(
                compare_values(name, candidate_a, candidate_b, compatibility)
            )

        warnings = self._deduplicate_warnings(warnings)
        if any(metric.comparison_status == "COMPARABLE" for metric in metrics):
            status = "OK_WITH_WARNINGS" if warnings else "OK"
        elif metrics and all(
            metric.comparison_status == "INSUFFICIENT_EVIDENCE" for metric in metrics
        ):
            status = "INSUFFICIENT_EVIDENCE"
        elif warnings:
            status = "OK_WITH_WARNINGS"
        else:
            status = "INSUFFICIENT_EVIDENCE"

        return ComparisonResult(
            status=status,
            question=request.question,
            companies=[company_a_id, company_b_id],
            metrics=metrics,
            warnings=warnings,
        )

    @staticmethod
    def _metric_names(
        evidence_a: AnswerEvidence,
        evidence_b: AnswerEvidence,
        requested: list[str],
    ) -> list[str]:
        if requested:
            return list(dict.fromkeys(_normalize_name(name) for name in requested))
        names_a = _available_names(evidence_a)
        names_b = _available_names(evidence_b)
        return sorted(names_a | names_b)

    @staticmethod
    def _select(
        evidence: AnswerEvidence, name: str, period: str | None, trend_mode: bool
    ) -> ComparisonEvidence | str | None:
        if trend_mode:
            return ComparisonAgent._select_trend(evidence, name, period)

        computed = [
            item
            for item in evidence.computed_metrics
            if item.company_id == evidence.company_id
            and _normalize_name(item.metric_name) == name
            and (period is None or item.period == period)
        ]
        if computed:
            if len(computed) != 1:
                return "multiple calculated values match the requested period"
            metric = computed[0]
            if metric.quality_status != "VALID" or metric.value is None:
                return "calculated metric is not valid"
            referenced_facts = {
                fact.fact_id: fact
                for fact in evidence.facts
                if fact.company_id == evidence.company_id
            }
            if not metric.input_fact_refs:
                return "calculated metric has no source fact references"
            if any(reference not in referenced_facts for reference in metric.input_fact_refs):
                return "calculated metric source facts are missing from the evidence bundle"
            inputs = [
                referenced_facts[reference]
                for reference in metric.input_fact_refs
            ]
            return ComparisonEvidence(
                company_id=evidence.company_id,
                company_name=evidence.company_name,
                value=metric.value,
                period=metric.period,
                currency=metric.currency,
                unit=metric.unit,
                display_unit=metric.display_unit,
                metric_definition=metric.metric_name,
                formula_version=metric.formula_version,
                evidence_refs=[metric.metric_id, *metric.input_fact_refs],
                source_references=_dedupe_sources(
                    fact.source_reference for fact in inputs
                ),
            )

        facts = [
            fact
            for fact in evidence.facts
            if fact.company_id == evidence.company_id
            and _normalize_name(fact.canonical_line_item) == name
            and (period is None or fact.fiscal_year == period)
        ]
        if len(facts) > 1:
            return "multiple reported values match the requested period"
        if not facts:
            return None
        fact = facts[0]
        return ComparisonEvidence(
            company_id=evidence.company_id,
            company_name=fact.company_name,
            value=fact.value,
            period=fact.fiscal_year,
            currency=fact.currency,
            unit=fact.unit,
            metric_definition=fact.metric_definition,
            evidence_refs=[fact.fact_id],
            source_references=[fact.source_reference],
        )

    @staticmethod
    def _select_trend(
        evidence: AnswerEvidence, name: str, period: str | None
    ) -> ComparisonEvidence | str | None:
        trends = [
            trend for trend in evidence.trends
            if trend.company_id == evidence.company_id
            and _normalize_name(trend.metric_name) == name
            and (period is None or trend.to_period == period)
        ]
        if len(trends) > 1:
            return "multiple trend results match the requested period"
        if not trends:
            return None
        trend = trends[0]
        if trend.quality_status != "VALID" or trend.delta is None:
            return "trend result is not valid"

        supported_metrics = [
            metric for metric in evidence.computed_metrics
            if metric.company_id == evidence.company_id
            and metric.metric_id in trend.supporting_refs
        ]
        by_period = {metric.period: metric for metric in supported_metrics}
        before = by_period.get(trend.from_period)
        after = by_period.get(trend.to_period)
        if (
            before is None or after is None
            or before.value is None or after.value is None
            or before.quality_status != "VALID"
            or after.quality_status != "VALID"
        ):
            return "trend lacks valid formula-versioned supporting metrics"
        if before.formula_version != after.formula_version:
            return "trend supporting metrics use different formula versions"

        referenced_ids = list(dict.fromkeys([
            trend.trend_id,
            *trend.supporting_refs,
            *before.input_fact_refs,
            *after.input_fact_refs,
        ]))
        referenced_facts = {
            fact.fact_id: fact
            for fact in evidence.facts
            if fact.company_id == evidence.company_id
        }
        source_fact_refs = list(dict.fromkeys([
            *before.input_fact_refs,
            *after.input_fact_refs,
        ]))
        if not source_fact_refs or any(
            reference not in referenced_facts for reference in source_fact_refs
        ):
            return "trend source facts are missing from the evidence bundle"
        source_facts = [
            referenced_facts[reference]
            for reference in source_fact_refs
        ]
        return ComparisonEvidence(
            company_id=evidence.company_id,
            company_name=evidence.company_name,
            value=trend.delta,
            period=f"{trend.from_period} to {trend.to_period}",
            currency=after.currency,
            unit=after.unit,
            display_unit=after.display_unit,
            metric_definition=trend.metric_name,
            formula_version=after.formula_version,
            evidence_refs=referenced_ids,
            source_references=_dedupe_sources(
                fact.source_reference for fact in source_facts
            ),
        )

    @staticmethod
    def _metadata(
        evidence: AnswerEvidence, name: str, period: str | None, trend_mode: bool
    ) -> tuple[object, ...] | None:
        if trend_mode:
            trends = [
                trend for trend in evidence.trends
                if trend.company_id == evidence.company_id
                and _normalize_name(trend.metric_name) == name
                and (period is None or trend.to_period == period)
            ]
            if len(trends) != 1:
                return None
            trend = trends[0]
            supported = {
                metric.period: metric
                for metric in evidence.computed_metrics
                if metric.company_id == evidence.company_id
                and metric.metric_id in trend.supporting_refs
            }
            before = supported.get(trend.from_period)
            after = supported.get(trend.to_period)
            if before is None or after is None:
                return None
            return (
                (trend.from_period, trend.to_period),
                (
                    (before.period_start, before.period_end),
                    (after.period_start, after.period_end),
                ),
                after.fiscal_year_end,
            )

        computed = [
            item for item in evidence.computed_metrics
            if item.company_id == evidence.company_id
            and _normalize_name(item.metric_name) == name
            and (period is None or item.period == period)
        ]
        if computed:
            item = computed[0]
            return (
                (item.period,),
                ((item.period_start, item.period_end),),
                item.fiscal_year_end,
            )
        facts = [
            item for item in evidence.facts
            if item.company_id == evidence.company_id
            and _normalize_name(item.canonical_line_item) == name
            and (period is None or item.fiscal_year == period)
        ]
        if facts:
            item = facts[0]
            return (
                (item.fiscal_year,),
                ((item.period_start, item.period_end),),
                item.fiscal_year_end,
            )
        return None

    @staticmethod
    def _period_range_match(
        a: tuple[object, ...] | None, b: tuple[object, ...] | None
    ) -> bool | None:
        if a is None or b is None:
            return None
        labels_a, ranges_a, _ = a
        labels_b, ranges_b, _ = b
        if labels_a != labels_b or len(ranges_a) != len(ranges_b):
            return False
        if any(
            start_a is None or end_a is None or start_b is None or end_b is None
            for (start_a, end_a), (start_b, end_b) in zip(ranges_a, ranges_b)
        ):
            return None
        return ranges_a == ranges_b

    @staticmethod
    def _fiscal_year_end_match(
        a: tuple[object, ...] | None, b: tuple[object, ...] | None
    ) -> bool | None:
        if a is None or b is None:
            return None
        end_a, end_b = a[2], b[2]
        if end_a is None or end_b is None:
            return None
        return end_a == end_b

    @staticmethod
    def _is_non_monetary(a: ComparisonEvidence, b: ComparisonEvidence) -> bool:
        units = {(a.display_unit or a.unit or "").lower(),
                 (b.display_unit or b.unit or "").lower()}
        return bool(units) and units <= NON_MONETARY_UNITS

    @staticmethod
    def _upstream_warnings(
        evidence_a: AnswerEvidence, evidence_b: AnswerEvidence
    ) -> list[ComparisonWarning]:
        warnings = []
        for evidence in (evidence_a, evidence_b):
            foreign_facts = [
                fact.fact_id for fact in evidence.facts
                if fact.company_id != evidence.company_id
            ]
            foreign_metrics = [
                metric.metric_id for metric in evidence.computed_metrics
                if metric.company_id != evidence.company_id
            ]
            if foreign_facts or foreign_metrics:
                warnings.append(
                    ComparisonWarning(
                        code="COMPANY_IDENTITY_MISMATCH",
                        message=(
                            f"{evidence.company_id}: ignored evidence assigned to "
                            f"another company ({', '.join([*foreign_facts, *foreign_metrics])})."
                        ),
                        company_ids=[evidence.company_id],
                    )
                )
            for warning in evidence.warnings:
                warnings.append(
                    ComparisonWarning(
                        code=warning.split(":", 1)[0],
                        message=f"{evidence.company_id}: {warning}",
                        company_ids=[evidence.company_id],
                    )
                )
            for metric in evidence.computed_metrics:
                if metric.company_id != evidence.company_id:
                    continue
                for warning in metric.warnings:
                    warnings.append(
                        ComparisonWarning(
                            code=warning.split(":", 1)[0],
                            message=f"{evidence.company_id}: {warning}",
                            metric_name=metric.metric_name,
                            company_ids=[evidence.company_id],
                        )
                    )
        return warnings

    @staticmethod
    def _deduplicate_warnings(
        warnings: list[ComparisonWarning],
    ) -> list[ComparisonWarning]:
        seen: set[tuple[str, str, str | None, tuple[str, ...]]] = set()
        result = []
        for warning in warnings:
            key = (
                warning.code,
                warning.message,
                warning.metric_name,
                tuple(warning.company_ids),
            )
            if key not in seen:
                result.append(warning)
                seen.add(key)
        return result


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _available_names(evidence: AnswerEvidence) -> set[str]:
    return {
        *(
            _normalize_name(item.metric_name)
            for item in evidence.computed_metrics
            if item.company_id == evidence.company_id
        ),
        *(
            _normalize_name(item.canonical_line_item)
            for item in evidence.facts
            if item.company_id == evidence.company_id
        ),
        *(
            _normalize_name(item.metric_name)
            for item in evidence.trends
            if item.company_id == evidence.company_id
        ),
    }


def _question_requests_trend(question: str) -> bool:
    normalized = question.lower()
    return any(
        word in normalized
        for word in ("improv", "trend", "changed", "change", "declin", "increased", "grew")
    )


def _dedupe_sources(sources: Iterable[SourceReference]) -> list[SourceReference]:
    unique: dict[str, SourceReference] = {}
    for source in sources:
        unique.setdefault(source.source_id, source)
    return list(unique.values())

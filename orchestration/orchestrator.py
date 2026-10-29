"""Application workflow connecting retrieval, analysis, comparison, and narrative."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from agents.comparison import ComparisonAgent, ComparisonRequest
from agents.narrative_qa.agent import NarrativeQAAgent
from agents.narrative_qa.citation_validator import ValidationResult
from agents.retrieval_grounding.agent import RetrievalGroundingAgent
from agents.trend_red_flag.agent import TrendRedFlagAgent
from models.schemas import AnswerEvidence
from orchestration.errors import (
    CitationValidationError,
    OrchestrationInputError,
)
from orchestration.router import QuestionRouter, RoutedQuestion
from orchestration.schemas import (
    OrchestrationRequest,
    OrchestrationResponse,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """Route a question through only the evidence and agents it requires."""

    def __init__(
        self,
        retrieval_agent: RetrievalGroundingAgent,
        *,
        trend_agent: TrendRedFlagAgent | None = None,
        comparison_agent: ComparisonAgent | None = None,
        narrative_agent: NarrativeQAAgent | None = None,
        question_router: QuestionRouter | None = None,
    ) -> None:
        self.retrieval_agent = retrieval_agent
        self.trend_agent = trend_agent or TrendRedFlagAgent()
        self.comparison_agent = comparison_agent or ComparisonAgent()
        self.narrative_agent = narrative_agent or NarrativeQAAgent()
        self.question_router = question_router or QuestionRouter()

    def answer(self, request: OrchestrationRequest) -> OrchestrationResponse:
        routed = self.question_router.route(
            request.question,
            request.period,
            has_comparison_company=request.comparison_company_id is not None,
        )
        if routed.comparison:
            return self._answer_comparison(request, routed)
        return self._answer_single_company(request, routed)

    def _answer_single_company(
        self, request: OrchestrationRequest, routed: RoutedQuestion
    ) -> OrchestrationResponse:
        evidence = self._retrieve_and_analyse(
            request.question,
            request.company_id,
            request.period,
            routed,
        )
        answer = self.narrative_agent.answer(evidence)
        validation = self.narrative_agent.citation_validator.validate(
            answer, evidence
        )
        self._log_request(
            request,
            routed,
            [evidence],
            warning_codes=_evidence_warning_codes([evidence]),
            citation_result=validation,
        )
        if not validation.valid:
            raise CitationValidationError(
                "The final answer failed citation validation: "
                + "; ".join(validation.errors)
            )
        status = (
            "INSUFFICIENT_EVIDENCE"
            if evidence.status == "INSUFFICIENT_EVIDENCE"
            else "OK_WITH_WARNINGS" if evidence.warnings else "OK"
        )
        return OrchestrationResponse(
            request_id=request.request_id,
            intent=routed.intent,
            status=status,
            answer=answer,
            evidence=[evidence],
            citation_validation="VALID",
        )

    def _answer_comparison(
        self, request: OrchestrationRequest, routed: RoutedQuestion
    ) -> OrchestrationResponse:
        comparison_company_id = request.comparison_company_id
        if comparison_company_id is None:
            raise OrchestrationInputError(
                "A comparison question requires comparison_company_id"
            )

        evidence_a = self._retrieve_and_analyse(
            request.question,
            request.company_id,
            request.period,
            routed,
        )
        evidence_b = self._retrieve_and_analyse(
            request.question,
            comparison_company_id,
            request.period,
            routed,
        )
        comparison = self.comparison_agent.compare(
            evidence_a,
            evidence_b,
            ComparisonRequest(
                question=request.question,
                metric_names=list(routed.comparison_metrics),
                period=request.period,
                compare_trends=routed.query_intent.wants_trends,
            ),
        )
        upstream_insufficient = any(
            evidence.status == "INSUFFICIENT_EVIDENCE"
            for evidence in (evidence_a, evidence_b)
        )
        answer = self.narrative_agent.answer_comparison(
            comparison,
            upstream_insufficient=upstream_insufficient,
        )
        validation = self.narrative_agent.citation_validator.validate_comparison(
            answer, comparison
        )
        warning_codes = _evidence_warning_codes([evidence_a, evidence_b])
        warning_codes.extend(warning.code for warning in comparison.warnings)
        warning_codes = list(dict.fromkeys(warning_codes))
        self._log_request(
            request,
            routed,
            [evidence_a, evidence_b],
            warning_codes=warning_codes,
            citation_result=validation,
        )
        if not validation.valid:
            raise CitationValidationError(
                "The final comparison answer failed citation validation: "
                + "; ".join(validation.errors)
            )
        status = (
            "INSUFFICIENT_EVIDENCE"
            if upstream_insufficient or comparison.status == "INSUFFICIENT_EVIDENCE"
            else comparison.status
        )
        return OrchestrationResponse(
            request_id=request.request_id,
            intent=routed.intent,
            status=status,
            answer=answer,
            evidence=[evidence_a, evidence_b],
            comparison=comparison,
            citation_validation="VALID",
        )

    def _retrieve_and_analyse(
        self,
        question: str,
        company_id: str,
        period: str | None,
        routed: RoutedQuestion,
    ) -> AnswerEvidence:
        evidence = self.retrieval_agent.retrieve_context(
            question,
            company_id,
            period,
            include_all_financial_facts=routed.include_all_financial_facts,
        )
        if (
            evidence.status == "INSUFFICIENT_EVIDENCE"
            or not routed.requires_trend_analysis
        ):
            return evidence

        ordered_periods = _chronological_periods(evidence.periods)
        analysis = self.trend_agent.analyse(
            company_id,
            evidence.facts,
            periods=ordered_periods or None,
            computed_metrics=evidence.computed_metrics,
        )
        requested_metrics = set(
            routed.comparison_metrics
            if routed.comparison
            else routed.query_intent.computed_metrics
        )
        include_all_results = routed.query_intent.wants_red_flags or (
            routed.comparison and not routed.comparison_metrics
        )
        selected_trends = [
            trend
            for trend in analysis.trends
            if routed.query_intent.wants_trends
            and (
                not requested_metrics
                or trend.metric_name in requested_metrics
            )
        ]
        selected_flags = (
            analysis.red_flags if routed.query_intent.wants_red_flags else []
        )
        selected_metrics = [
            metric
            for metric in analysis.computed_metrics
            if include_all_results
            or metric.metric_name in requested_metrics
            or any(
                metric.metric_id in trend.supporting_refs
                for trend in selected_trends
            )
        ]
        metrics_by_identity = {
            (metric.metric_name, metric.period): metric
            for metric in evidence.computed_metrics
            if include_all_results
            or metric.metric_name in requested_metrics
        }
        metrics_by_identity.update({
            (metric.metric_name, metric.period): metric
            for metric in selected_metrics
        })
        warnings = list(evidence.warnings)
        for item in [*selected_metrics, *selected_trends, *selected_flags]:
            warnings.extend(item.warnings)
        warnings = list(dict.fromkeys(warnings))
        return evidence.model_copy(
            update={
                "periods": analysis.periods or evidence.periods,
                "facts": analysis.facts,
                "computed_metrics": list(metrics_by_identity.values()),
                "trends": selected_trends,
                "red_flags": selected_flags,
                "warnings": warnings,
            }
        )

    @staticmethod
    def _log_request(
        request: OrchestrationRequest,
        routed: RoutedQuestion,
        evidence: Sequence[AnswerEvidence],
        *,
        warning_codes: list[str],
        citation_result: ValidationResult,
    ) -> None:
        logger.info(
            "Financial question orchestration completed",
            extra={
                "request_id": request.request_id,
                "company_id": request.company_id,
                "company_ids": [item.company_id for item in evidence],
                "question_intent": routed.intent.value,
                "retrieved_fact_ids": {
                    item.company_id: [fact.fact_id for fact in item.facts]
                    for item in evidence
                },
                "computed_metric_ids": {
                    item.company_id: [
                        metric.metric_id for metric in item.computed_metrics
                    ]
                    for item in evidence
                },
                "red_flag_ids": {
                    item.company_id: [flag.flag_id for flag in item.red_flags]
                    for item in evidence
                },
                "warning_codes": warning_codes,
                "citation_validation_result": (
                    "VALID" if citation_result.valid else "INVALID"
                ),
            },
        )


def _chronological_periods(periods: Sequence[str]) -> list[str]:
    indexed = list(enumerate(periods))

    def sort_key(item: tuple[int, str]) -> tuple[int, int, str]:
        index, period = item
        match = re.search(r"\d{4}", period)
        if match:
            return (0, int(match.group()), period)
        return (1, index, period)

    return [period for _, period in sorted(indexed, key=sort_key)]


def _evidence_warning_codes(evidence: Sequence[AnswerEvidence]) -> list[str]:
    codes: list[str] = []
    for bundle in evidence:
        for warning in bundle.warnings:
            codes.append(warning.split(":", 1)[0])
        for metric in bundle.computed_metrics:
            codes.extend(warning.split(":", 1)[0] for warning in metric.warnings)
        for flag in bundle.red_flags:
            codes.extend(warning.split(":", 1)[0] for warning in flag.warnings)
    return list(dict.fromkeys(codes))

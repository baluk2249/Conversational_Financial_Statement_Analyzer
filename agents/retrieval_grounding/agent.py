"""Evidence orchestration; numeric truth remains in structured/deterministic tools."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from finance.ratios import RatioCalculator
from agents.retrieval_grounding.config import METRIC_ALIASES
from models.schemas import (
    AnswerEvidence,
    ComputedMetric,
    InsufficientEvidence,
)
from retrieval.interfaces import (
    NarrativeVectorStore,
    NarrativeStoreUnavailableError,
    StructuredFinancialStore,
    TrendAndRedFlagProvider,
)
from agents.retrieval_grounding.query_parser import DeterministicQueryParser

logger = logging.getLogger(__name__)


class RetrievalGroundingAgent:
    """Build company-scoped evidence bundles for downstream narrative agents."""

    def __init__(
        self,
        structured_store: StructuredFinancialStore,
        narrative_store: NarrativeVectorStore | None = None,
        result_provider: TrendAndRedFlagProvider | None = None,
        query_parser: DeterministicQueryParser | None = None,
        ratio_calculator: RatioCalculator | None = None,
    ) -> None:
        self.structured_store = structured_store
        self.narrative_store = narrative_store
        self.result_provider = result_provider or structured_store  # type: ignore[assignment]
        self.query_parser = query_parser or DeterministicQueryParser()
        self.ratio_calculator = ratio_calculator or RatioCalculator(structured_store)

    def retrieve_context(
        self,
        question: str,
        company_id: str,
        period: str | None = None,
        top_k: int = 5,
        *,
        include_all_financial_facts: bool = False,
    ) -> AnswerEvidence:
        intent = self.query_parser.parse(question, period)
        requested_periods = list(intent.periods) or None
        query_periods = self._include_prior_period(requested_periods) if intent.wants_trends else requested_periods
        fact_metrics = list(intent.fact_metrics)
        if include_all_financial_facts:
            fact_metrics = list(dict.fromkeys([*fact_metrics, *METRIC_ALIASES]))
        statement_filter = (
            None if include_all_financial_facts else intent.statement_type
        )
        facts = self.structured_store.get_facts(
            company_id, fact_metrics, query_periods, statement_filter,
        )
        facts = [
            fact for fact in facts
            if fact.company_id == company_id
            and fact.canonical_line_item in fact_metrics
            and (query_periods is None or fact.fiscal_year in query_periods)
            and (statement_filter is None or fact.statement_type == statement_filter)
        ]
        available_periods = sorted({fact.fiscal_year for fact in facts})
        periods_for_metrics = query_periods
        if intent.computed_metrics and periods_for_metrics is None:
            periods_for_metrics = available_periods if intent.wants_trends else available_periods[-1:]

        computed = self._get_computed(intent.computed_metrics, company_id, periods_for_metrics)
        computed = [
            metric for metric in computed
            if metric.company_id == company_id
            and metric.metric_name in intent.computed_metrics
            and (periods_for_metrics is None or metric.period in periods_for_metrics)
        ]
        for metric_name in intent.computed_metrics:
            metric_periods = periods_for_metrics or [fact.fiscal_year for fact in facts]
            for metric_period in dict.fromkeys(metric_periods):
                if not any(item.metric_name == metric_name and item.period == metric_period for item in computed):
                    computed.append(self.ratio_calculator.calculate(company_id, metric_name, metric_period))

        warnings = self._compatibility_warnings(facts)
        for metric in computed:
            warnings.extend(metric.warnings)
        warnings = list(dict.fromkeys(warnings))

        trends = []
        red_flags = []
        if intent.wants_trends and self.result_provider:
            trends = [trend for trend in self.result_provider.get_trends(
                company_id,
                metrics=intent.computed_metrics or None,
                periods=query_periods,
            ) if trend.company_id == company_id
                and (requested_periods is None or trend.to_period in requested_periods)
                and (not intent.computed_metrics or trend.metric_name in intent.computed_metrics)]
        if intent.wants_red_flags and self.result_provider:
            red_flags = [flag for flag in self.result_provider.get_red_flags(
                company_id, periods=query_periods,
            ) if flag.company_id == company_id
                and (requested_periods is None or flag.period in requested_periods)]

        narrative = []
        narrative_unavailable = False
        if intent.needs_narrative and self.narrative_store is not None:
            search_periods = query_periods or [None]
            seen: set[str] = set()
            for search_period in search_periods:
                try:
                    chunks = self.narrative_store.search(
                        query=question,
                        company_id=company_id,
                        period=search_period,
                        document_type=intent.document_type,
                        top_k=top_k,
                    )
                except NarrativeStoreUnavailableError:
                    logger.exception(
                        "Narrative retrieval unavailable; retaining structured evidence"
                    )
                    narrative_unavailable = True
                    break
                for chunk in chunks:
                    if chunk.company_id == company_id and chunk.chunk_id not in seen:
                        if search_period is None or chunk.period == search_period:
                            narrative.append(chunk)
                            seen.add(chunk.chunk_id)

        company_name = next((fact.company_name for fact in facts), None)
        if company_name is None and narrative:
            company_name = narrative[0].company_name
        output_periods = query_periods or available_periods
        missing_inputs = self._missing_inputs(intent.computed_metrics, computed)
        if narrative_unavailable:
            warnings.append("NARRATIVE_RETRIEVAL_UNAVAILABLE")
            warnings = list(dict.fromkeys(warnings))
        no_requested_fact = bool(fact_metrics) and (
            not facts
            if include_all_financial_facts
            else not (facts or computed or narrative)
        )
        unsupported_exact = bool(intent.requested_item and not intent.needs_narrative and not facts and not computed)
        qualitative_empty = intent.needs_narrative and not (facts or computed or trends or red_flags or narrative)
        unavailable_requested_metric = any(
            metric.metric_name in intent.computed_metrics and metric.value is None
            for metric in computed
        )
        insufficient = (
            no_requested_fact or unsupported_exact or bool(missing_inputs)
            or unavailable_requested_metric or qualitative_empty
        )

        insufficient_detail = None
        status = "OK"
        if insufficient:
            status = "INSUFFICIENT_EVIDENCE"
            requested_item = intent.requested_item or (intent.computed_metrics[0].replace("_", " ").title() if intent.computed_metrics else "requested financial evidence")
            insufficient_detail = InsufficientEvidence(
                requested_item=requested_item,
                missing_inputs=(
                    missing_inputs or fact_metrics
                    if (no_requested_fact or unsupported_exact or unavailable_requested_metric)
                    else []
                ),
                available_evidence=[fact.fact_id for fact in facts] + [metric.metric_id for metric in computed if metric.value is not None],
            )

        logger.info(
            "Assembled financial evidence",
            extra={"company_id": company_id, "periods": output_periods, "fact_count": len(facts),
                   "metric_count": len(computed), "narrative_count": len(narrative), "status": status},
        )
        return AnswerEvidence(
            status=status,
            question=question,
            company_id=company_id,
            company_name=company_name,
            periods=output_periods,
            facts=facts,
            computed_metrics=computed,
            trends=trends,
            red_flags=red_flags,
            narrative_evidence=narrative,
            warnings=warnings,
            insufficient_evidence=insufficient_detail,
        )

    def _get_computed(
        self, metrics: Sequence[str], company_id: str, periods: Sequence[str] | None,
    ) -> list[ComputedMetric]:
        if not metrics:
            return []
        return self.structured_store.get_computed_metrics(company_id, metrics, periods)

    @staticmethod
    def _compatibility_warnings(facts: Sequence[object]) -> list[str]:
        currencies = {getattr(fact, "currency") for fact in facts if getattr(fact, "currency", None)}
        units = {getattr(fact, "unit") for fact in facts if getattr(fact, "unit", None)}
        fiscal_year_ends = {getattr(fact, "fiscal_year_end") for fact in facts if getattr(fact, "fiscal_year_end", None)}
        period_lengths = {
            (fact.period_end - fact.period_start).days + 1
            for fact in facts if getattr(fact, "period_start", None) and getattr(fact, "period_end", None)
        }
        definitions: dict[str, set[str]] = {}
        for fact in facts:
            definition = getattr(fact, "metric_definition", None)
            if definition:
                definitions.setdefault(fact.canonical_line_item, set()).add(definition)
        warnings: list[str] = []
        if len(currencies) > 1:
            warnings.append("CURRENCY_MISMATCH")
        if len(units) > 1:
            warnings.append("UNIT_MISMATCH")
        if period_lengths and max(period_lengths) - min(period_lengths) > 1:
            warnings.append("FISCAL_PERIOD_MISMATCH")
        if len(fiscal_year_ends) > 1:
            warnings.append("FISCAL_YEAR_END_MISMATCH")
        if any(len(values) > 1 for values in definitions.values()):
            warnings.append("METRIC_DEFINITION_MISMATCH")
        return warnings

    @staticmethod
    def _missing_inputs(metrics: Sequence[str], computed: Sequence[ComputedMetric]) -> list[str]:
        missing: list[str] = []
        requested = set(metrics)
        for result in computed:
            if result.metric_name in requested and result.value is None:
                missing.extend(
                    warning.removeprefix("MISSING_INPUT:")
                    for warning in result.warnings
                    if warning.startswith("MISSING_INPUT:")
                )
                if not result.warnings:
                    missing.append(result.metric_name)
        return list(dict.fromkeys(missing))

    @staticmethod
    def _include_prior_period(periods: list[str] | None) -> list[str] | None:
        if not periods:
            return periods
        expanded = list(periods)
        for requested_period in periods:
            match = re.fullmatch(r"FY\s?(\d{4})", requested_period, re.IGNORECASE)
            if match:
                prior = f"FY{int(match.group(1)) - 1}"
                if prior not in expanded:
                    expanded.append(prior)
        return expanded
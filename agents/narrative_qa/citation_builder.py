"""Deterministic citations and evidence lineage for narrative answers."""

from __future__ import annotations

from collections.abc import Sequence

from models.schemas import (
    AnswerEvidence,
    ComputedMetric,
    DocumentChunk,
    FinancialFact,
    RedFlagResult,
    TrendResult,
)


class CitationBuilder:
    """Build citations exclusively from source locators in ``AnswerEvidence``."""

    def fact_citation(self, fact: FinancialFact) -> str:
        return fact.source_reference.citation

    def metric_citations(
        self, metric: ComputedMetric, evidence: AnswerEvidence
    ) -> list[str]:
        facts_by_id = {
            fact.fact_id: fact
            for fact in evidence.facts
            if fact.company_id == metric.company_id == evidence.company_id
        }
        return self._fact_citations(metric.input_fact_refs, facts_by_id)

    def trend_citations(
        self, trend: TrendResult, evidence: AnswerEvidence
    ) -> list[str]:
        return self._reference_citations(trend.supporting_refs, evidence)

    def flag_citations(
        self, flag: RedFlagResult, evidence: AnswerEvidence
    ) -> list[str]:
        return self._reference_citations(flag.supporting_refs, evidence)

    def narrative_citation(self, chunk: DocumentChunk) -> str:
        return chunk.source_reference.citation

    def _reference_citations(
        self, references: Sequence[str], evidence: AnswerEvidence
    ) -> list[str]:
        facts_by_id = {
            fact.fact_id: fact
            for fact in evidence.facts
            if fact.company_id == evidence.company_id
        }
        metrics_by_id = {
            metric.metric_id: metric
            for metric in evidence.computed_metrics
            if metric.company_id == evidence.company_id
        }
        trends_by_id = {
            trend.trend_id: trend
            for trend in evidence.trends
            if trend.company_id == evidence.company_id
        }
        citations: list[str] = []
        pending = list(references)
        visited: set[str] = set()
        while pending:
            reference = pending.pop()
            if reference in visited:
                continue
            visited.add(reference)
            if reference in facts_by_id:
                citations.append(self.fact_citation(facts_by_id[reference]))
            elif reference in metrics_by_id:
                pending.extend(metrics_by_id[reference].input_fact_refs)
            elif reference in trends_by_id:
                pending.extend(trends_by_id[reference].supporting_refs)
        return list(dict.fromkeys(citations))

    @staticmethod
    def _fact_citations(
        fact_ids: Sequence[str], facts_by_id: dict[str, FinancialFact]
    ) -> list[str]:
        return list(
            dict.fromkeys(
                facts_by_id[fact_id].source_reference.citation
                for fact_id in fact_ids
                if fact_id in facts_by_id
            )
        )

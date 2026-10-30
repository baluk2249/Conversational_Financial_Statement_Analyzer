"""Compose the existing agents with persistent stores and upload processing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from agents.comparison import ComparisonAgent
from agents.comparison.schemas import ComparisonRequest, ComparisonResult
from agents.narrative_qa.agent import NarrativeQAAgent
from agents.narrative_qa.citation_builder import CitationBuilder
from agents.retrieval_grounding.agent import RetrievalGroundingAgent
from agents.retrieval_grounding.config import (
    COMPUTED_METRIC_INPUTS,
    METRIC_ALIASES,
)
from agents.trend_red_flag.agent import TrendRedFlagAgent
from finance.ratios import calculate_ratio
from ingestion import parse_document
from models.schemas import AnswerEvidence, ComputedMetric, DocumentChunk
from orchestration import OrchestrationRequest, Orchestrator
from retrieval.faiss_store import FaissNarrativeStore
from retrieval.interfaces import NarrativeStoreUnavailableError
from retrieval.structured_store import SQLiteFinancialStore

logger = logging.getLogger(__name__)

RATIO_SUMMARY_SECTIONS = {
    "Liquidity": (
        "current_ratio",
        "quick_ratio",
        "operating_cash_flow_ratio",
    ),
    "Leverage": (
        "debt_to_equity",
        "debt_to_assets",
        "interest_coverage",
    ),
    "Profitability": (
        "gross_margin",
        "operating_margin",
        "net_margin",
        "roa",
        "roe",
    ),
    "Efficiency": ("dso", "inventory_days", "asset_turnover"),
}


class LazyFaissNarrativeStore:
    """Avoid downloading/loading an embedding model during API startup."""

    def __init__(self, persist_directory: str | Path) -> None:
        self.persist_directory = Path(persist_directory)
        self._store: FaissNarrativeStore | None = None
        self._error: NarrativeStoreUnavailableError | None = None

    @property
    def health_status(self) -> str:
        if self._error is not None:
            return "unavailable"
        return "healthy" if self._store is not None else "not_initialized"

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        self._get_store().add_chunks(chunks)

    def search(
        self,
        query: str,
        company_id: str,
        period: str | None = None,
        document_type: str | None = None,
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        return self._get_store().search(
            query, company_id, period, document_type, top_k
        )

    def _get_store(self) -> FaissNarrativeStore:
        if self._store is not None:
            return self._store
        if self._error is not None:
            raise self._error
        try:
            self._store = FaissNarrativeStore(
                persist_directory=self.persist_directory
            )
        except NarrativeStoreUnavailableError as exc:
            self._error = exc
            raise
        return self._store


class CompanyNotFoundError(LookupError):
    """Raised when an operation references an unknown company."""


class PeriodNotFoundError(LookupError):
    """Raised when a requested fiscal period is not available."""


class AnalyzerService:
    """Application boundary for ingestion, retrieval, analysis, and comparison."""

    def __init__(
        self,
        structured_store: SQLiteFinancialStore,
        narrative_store: LazyFaissNarrativeStore,
        *,
        narrative_agent: NarrativeQAAgent | None = None,
    ) -> None:
        self.structured_store = structured_store
        self.narrative_store = narrative_store
        self.retrieval_agent = RetrievalGroundingAgent(
            structured_store, narrative_store
        )
        self.trend_agent = TrendRedFlagAgent()
        self.comparison_agent = ComparisonAgent()
        self.narrative_agent = narrative_agent or _configured_narrative_agent()
        self.orchestrator = Orchestrator(
            self.retrieval_agent,
            trend_agent=self.trend_agent,
            comparison_agent=self.comparison_agent,
            narrative_agent=self.narrative_agent,
        )

    @classmethod
    def from_environment(cls) -> AnalyzerService:
        database_path = os.environ.get(
            "FINANCIAL_DATABASE_PATH", ".data/financial-facts.sqlite3"
        )
        faiss_path = os.environ.get("FAISS_INDEX_DIRECTORY", ".data/faiss")
        return cls(
            SQLiteFinancialStore(database_path),
            LazyFaissNarrativeStore(faiss_path),
        )

    def health(self) -> dict[str, Any]:
        components: dict[str, str] = {"api": "healthy"}
        try:
            self.structured_store.check_health()
            components["structured_store"] = "healthy"
        except Exception:
            logger.exception("Structured store health check failed")
            components["structured_store"] = "unavailable"
        components["faiss"] = self.narrative_store.health_status
        components["orchestrator"] = "healthy"
        status = (
            "healthy"
            if components["structured_store"] == "healthy"
            else "degraded"
        )
        return {"status": status, "components": components}

    def list_companies(self) -> list[dict[str, str]]:
        return self.structured_store.list_companies()

    def get_periods(self, company_id: str) -> list[str]:
        self._require_company(company_id)
        return self.structured_store.get_periods(company_id)

    def ingest_document(
        self,
        filename: str,
        content: bytes,
        *,
        company_name: str | None = None,
        company_id: str | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        result = parse_document(
            filename,
            content,
            company_name=company_name,
            company_id=company_id,
            period=period,
        )
        for fact in result.facts:
            self.structured_store.add_fact(fact)

        company_registry: dict[str, dict[str, Any]] = {}
        for fact in result.facts:
            entry = company_registry.setdefault(
                fact.company_id,
                {"company_name": fact.company_name, "periods": set()},
            )
            entry["periods"].add(fact.fiscal_year)
        for chunk in result.chunks:
            entry = company_registry.setdefault(
                chunk.company_id,
                {"company_name": chunk.company_name, "periods": set()},
            )
            if chunk.period:
                entry["periods"].add(chunk.period)
        for registered_id, entry in company_registry.items():
            self.structured_store.register_company(
                registered_id,
                entry["company_name"],
                sorted(entry["periods"]),
            )

        warnings = list(result.warnings)
        if result.chunks:
            try:
                self.narrative_store.add_chunks(result.chunks)
            except NarrativeStoreUnavailableError:
                logger.exception("FAISS indexing failed; structured facts were retained")
                warnings.append("NARRATIVE_INDEX_UNAVAILABLE")
        return {
            "status": "PROCESSED",
            "source_id": result.source_id,
            "source_file": result.source_file,
            "company_ids": sorted(
                {fact.company_id for fact in result.facts}
                | {chunk.company_id for chunk in result.chunks}
            ),
            "company_names": sorted(
                {fact.company_name for fact in result.facts}
                | {chunk.company_name for chunk in result.chunks}
            ),
            "periods": result.periods,
            "statement_types": result.statement_types,
            "facts_stored": len(result.facts),
            "narrative_chunks": len(result.chunks),
            "warnings": list(dict.fromkeys(warnings)),
        }

    def ask(
        self,
        company_id: str,
        question: str,
        period: str | None = None,
        comparison_company_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_company(company_id)
        if period is not None:
            self._require_period(company_id, period)
        if comparison_company_id:
            self._require_company(comparison_company_id)
        response = self.orchestrator.answer(
            OrchestrationRequest(
                company_id=company_id,
                comparison_company_id=comparison_company_id,
                question=question,
                period=period,
            )
        )
        citations = _response_citations(response)
        payload = response.model_dump(mode="json")
        payload["citations"] = citations
        return payload

    def get_ratios(
        self, company_id: str, period: str | None = None
    ) -> dict[str, Any]:
        periods = self.get_periods(company_id)
        if period is not None:
            self._require_period(company_id, period)
            selected_periods = [period]
        else:
            selected_periods = periods[-1:]
        facts = self.structured_store.get_facts(
            company_id, list(METRIC_ALIASES), periods
        )
        metrics = [
            calculate_ratio(
                facts, company_id, metric_name, selected_period
            )
            for selected_period in selected_periods
            for metric_name in COMPUTED_METRIC_INPUTS
        ]
        evidence = AnswerEvidence(
            question="Financial ratio dashboard",
            company_id=company_id,
            periods=selected_periods,
            facts=facts,
            computed_metrics=metrics,
        )
        citations = CitationBuilder()
        serialized_metrics = []
        for metric in metrics:
            serialized = metric.model_dump(mode="json")
            serialized["citations"] = citations.metric_citations(metric, evidence)
            serialized_metrics.append(serialized)
        return {
            "company_id": company_id,
            "periods": selected_periods,
            "metrics": serialized_metrics,
            "available_fact_count": len(facts),
        }

    def summarize_ratio_section(
        self,
        company_id: str,
        section: str,
        period: str | None = None,
    ) -> dict[str, Any]:
        metric_names = RATIO_SUMMARY_SECTIONS.get(section)
        if metric_names is None:
            raise ValueError(f"Unsupported ratio summary section: {section}")
        result = self.get_ratios(company_id, period)
        selected_metrics = [
            ComputedMetric.model_validate(
                {key: value for key, value in metric.items() if key != "citations"}
            )
            for metric in result["metrics"]
            if metric["metric_name"] in metric_names
        ]
        input_fact_ids = {
            fact_id
            for metric in selected_metrics
            for fact_id in metric.input_fact_refs
        }
        facts = self.structured_store.get_facts(
            company_id, list(METRIC_ALIASES), self.get_periods(company_id)
        )
        supporting_facts = [
            fact for fact in facts if fact.fact_id in input_fact_ids
        ]
        warnings = list(
            dict.fromkeys(
                warning
                for metric in selected_metrics
                for warning in metric.warnings
            )
        )
        evidence = AnswerEvidence(
            question=f"Summarize the {section} financial ratios section.",
            company_id=company_id,
            company_name=supporting_facts[0].company_name
            if supporting_facts
            else None,
            periods=result["periods"],
            facts=supporting_facts,
            computed_metrics=selected_metrics,
            warnings=warnings,
        )
        return {
            "company_id": company_id,
            "periods": result["periods"],
            "section": section,
            "summary": self.narrative_agent.summarize(
                evidence, section=section
            ),
        }

    def get_red_flags(
        self, company_id: str, period: str | None = None
    ) -> dict[str, Any]:
        periods = self.get_periods(company_id)
        if period is not None:
            self._require_period(company_id, period)
        facts = self.structured_store.get_facts(
            company_id, list(METRIC_ALIASES), periods
        )
        analysis = self.trend_agent.analyse(
            company_id,
            facts,
            periods=periods,
        )
        flags = [
            flag for flag in analysis.red_flags
            if period is None or flag.period == period
        ]
        evidence = AnswerEvidence(
            question="Financial red-flag dashboard",
            company_id=company_id,
            periods=analysis.periods,
            facts=analysis.facts,
            computed_metrics=analysis.computed_metrics,
            trends=analysis.trends,
        )
        citation_builder = CitationBuilder()
        serialized_flags = []
        for flag in flags:
            serialized = flag.model_dump(mode="json")
            serialized["citations"] = citation_builder.flag_citations(
                flag, evidence
            )
            serialized_flags.append(serialized)
        return {
            "company_id": company_id,
            "periods": [period] if period else periods,
            "red_flags": serialized_flags,
        }

    def summarize_red_flags(
        self, company_id: str, period: str | None = None
    ) -> dict[str, Any]:
        periods = self.get_periods(company_id)
        if period is not None:
            self._require_period(company_id, period)
        facts = self.structured_store.get_facts(
            company_id, list(METRIC_ALIASES), periods
        )
        analysis = self.trend_agent.analyse(
            company_id,
            facts,
            periods=periods,
        )
        flags = [
            flag for flag in analysis.red_flags
            if period is None or flag.period == period
        ]
        evidence = AnswerEvidence(
            question="Summarize the red-flag indicators and their limitations.",
            company_id=company_id,
            periods=analysis.periods,
            facts=analysis.facts,
            computed_metrics=analysis.computed_metrics,
            trends=analysis.trends,
            red_flags=flags,
            warnings=list(
                dict.fromkeys(
                    warning for flag in flags for warning in flag.warnings
                )
            ),
        )
        return {
            "company_id": company_id,
            "periods": [period] if period else periods,
            "summary": self.narrative_agent.summarize(
                evidence, section="red flags"
            ),
        }

    def compare(
        self,
        company_a: str,
        company_b: str,
        *,
        metrics: list[str] | None = None,
        period_a: str | None = None,
        period_b: str | None = None,
    ) -> dict[str, Any]:
        self._require_company(company_a)
        self._require_company(company_b)
        if company_a == company_b:
            raise ValueError("Comparison companies must be different.")
        if period_a is not None:
            self._require_period(company_a, period_a)
        if period_b is not None:
            self._require_period(company_b, period_b)

        metric_names = metrics or list(COMPUTED_METRIC_INPUTS)
        question = "Compare " + ", ".join(
            item.replace("_", " ") for item in metric_names
        )
        evidence_a = self.retrieval_agent.retrieve_context(
            question,
            company_a,
            period_a,
            include_all_financial_facts=True,
        )
        evidence_b = self.retrieval_agent.retrieve_context(
            question,
            company_b,
            period_b,
            include_all_financial_facts=True,
        )
        result = self.comparison_agent.compare(
            evidence_a,
            evidence_b,
            ComparisonRequest(
                question=question,
                metric_names=metric_names,
            ),
        )
        payload = result.model_dump(mode="json")
        for source, serialized in zip(
            result.metrics, payload["metrics"], strict=True
        ):
            for attribute in ("company_a", "company_b"):
                evidence = getattr(source, attribute)
                if evidence is not None and serialized[attribute] is not None:
                    serialized[attribute]["citations"] = [
                        reference.citation
                        for reference in evidence.source_references
                    ]
        return payload

    def summarize_comparison(
        self,
        company_a: str,
        company_b: str,
        *,
        metrics: list[str] | None = None,
        period_a: str | None = None,
        period_b: str | None = None,
    ) -> dict[str, Any]:
        payload = self.compare(
            company_a,
            company_b,
            metrics=metrics,
            period_a=period_a,
            period_b=period_b,
        )
        for metric in payload["metrics"]:
            for company_key in ("company_a", "company_b"):
                evidence = metric.get(company_key)
                if evidence is not None:
                    evidence.pop("citations", None)
        result = ComparisonResult.model_validate(payload)
        return {
            "companies": result.companies,
            "status": result.status,
            "summary": self.narrative_agent.summarize_comparison(result),
        }

    def _require_company(self, company_id: str) -> None:
        if not any(item["company_id"] == company_id for item in self.list_companies()):
            raise CompanyNotFoundError(company_id)

    def _require_period(self, company_id: str, period: str) -> None:
        if period not in self.structured_store.get_periods(company_id):
            raise PeriodNotFoundError(period)


def _configured_narrative_agent() -> NarrativeQAAgent:
    if not os.environ.get("OPENAI_API_KEY"):
        return NarrativeQAAgent()
    from llm.providers import OpenAICompatibleClient

    return NarrativeQAAgent(
        llm_client=OpenAICompatibleClient.from_environment()
    )


def _response_citations(response: Any) -> list[str]:
    citations: list[str] = []
    for evidence in response.evidence:
        citations.extend(
            fact.source_reference.citation for fact in evidence.facts
        )
        citations.extend(
            chunk.source_reference.citation
            for chunk in evidence.narrative_evidence
        )
    if response.comparison is not None:
        for metric in response.comparison.metrics:
            for item in (metric.company_a, metric.company_b):
                if item is not None:
                    citations.extend(
                        source.citation for source in item.source_references
                    )
    return list(dict.fromkeys(citations))

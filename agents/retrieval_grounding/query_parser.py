"""Deterministic MVP question-to-evidence mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents.retrieval_grounding.config import COMPUTED_METRIC_ALIASES, COMPUTED_METRIC_INPUTS, METRIC_ALIASES


@dataclass(frozen=True)
class QueryIntent:
    fact_metrics: tuple[str, ...] = ()
    computed_metrics: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    needs_narrative: bool = False
    wants_trends: bool = False
    wants_red_flags: bool = False
    requested_item: str | None = None
    statement_type: str | None = None
    document_type: str | None = None


class DeterministicQueryParser:
    """Maps common terms to normalized names; it does not infer numeric values."""

    def parse(self, question: str, period: str | None = None) -> QueryIntent:
        normalized = re.sub(r"\s+", " ", question.casefold())
        computed = tuple(
            name for name, aliases in COMPUTED_METRIC_ALIASES.items()
            if any(self._contains(normalized, alias) for alias in aliases)
        )
        facts: list[str] = []
        for name, aliases in METRIC_ALIASES.items():
            if any(self._contains(normalized, alias) for alias in aliases):
                facts.append(name)
        for metric in computed:
            facts.extend(COMPUTED_METRIC_INPUTS[metric])
        if "profitability" in normalized or "profitability decline" in normalized:
            facts.extend(("revenue", "gross_profit", "operating_profit", "net_income"))
            computed += tuple(name for name in ("gross_margin", "operating_margin", "net_margin") if name not in computed)

        wants_red_flags = any(term in normalized for term in ("red flag", "warning sign", "risk flag"))
        wants_trends = any(term in normalized for term in ("trend", "declin", "increas", "decreas", "change", "over time", "period-over-period"))
        qualitative = any(
            term in normalized
            for term in (
                "why", "explain", "commentary", "reason", "cause", "discuss",
                "summary", "summarize", "summarise", "overview",
            )
        )
        if wants_red_flags:
            wants_trends = True
        if wants_trends and not wants_red_flags and not facts and not computed:
            computed = tuple(COMPUTED_METRIC_INPUTS)
            facts.extend(
                line_item
                for metric_name in computed
                for line_item in COMPUTED_METRIC_INPUTS[metric_name]
            )

        periods = list(dict.fromkeys(re.findall(r"\bFY\s?\d{4}\b", question, flags=re.IGNORECASE)))
        periods = [re.sub(r"\s+", "", item).upper() for item in periods]
        if period and period not in periods:
            periods.insert(0, period)
        elif period:
            periods = [period, *(item for item in periods if item != period)]

        topics = self._topics(normalized)
        if qualitative and not topics:
            topics = ("financial performance",)
        requested_item = self._requested_label(question, normalized)
        statement_type = None
        if "balance sheet" in normalized:
            statement_type = "balance_sheet"
        elif "cash flow statement" in normalized or "cashflow statement" in normalized:
            statement_type = "cash_flow_statement"
        elif "income statement" in normalized or "profit and loss" in normalized or "p&l" in normalized:
            statement_type = "income_statement"
        document_type = "annual_report" if "annual report" in normalized or "annual report" in normalized.replace("-", " ") else None
        return QueryIntent(
            fact_metrics=tuple(dict.fromkeys(facts)),
            computed_metrics=computed,
            topics=topics,
            periods=tuple(periods),
            needs_narrative=qualitative or wants_trends or wants_red_flags,
            wants_trends=wants_trends,
            wants_red_flags=wants_red_flags,
            requested_item=requested_item,
            statement_type=statement_type,
            document_type=document_type,
        )

    @staticmethod
    def _contains(text: str, phrase: str) -> bool:
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

    @staticmethod
    def _topics(text: str) -> tuple[str, ...]:
        topic_terms = {
            "profitability": ("profitability", "margin", "profit"),
            "working capital": ("working capital", "receivable", "inventory", "payable"),
            "liquidity": ("liquidity", "cash flow", "current ratio", "quick ratio"),
            "leverage": ("leverage", "debt", "borrowings", "interest"),
            "accounting policy": ("accounting policy", "accounting policies", "policy"),
            "costs": ("cost", "expense", "overhead"),
        }
        return tuple(topic for topic, terms in topic_terms.items() if any(term in text for term in terms))

    @staticmethod
    def _requested_label(question: str, normalized: str) -> str | None:
        if "ebitda" in normalized:
            return "EBITDA"
        for name, aliases in COMPUTED_METRIC_ALIASES.items():
            if any(DeterministicQueryParser._contains(normalized, alias) for alias in aliases):
                return name.replace("_", " ").title()
        for name, aliases in METRIC_ALIASES.items():
            if any(DeterministicQueryParser._contains(normalized, alias) for alias in aliases):
                return name.replace("_", " ").title()
        match = re.search(r"\b(?:what (?:was|is)|how much|give me)\s+(?:the\s+)?(.+?)(?:\s+in\s+FY\d{4})?[?\.]?$", question, re.IGNORECASE)
        if match:
            return match.group(1).strip().upper()
        return None
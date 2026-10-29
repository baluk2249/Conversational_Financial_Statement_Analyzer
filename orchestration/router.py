"""Deterministic question routing, separate from financial calculations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.retrieval_grounding.query_parser import (
    DeterministicQueryParser,
    QueryIntent,
)
from orchestration.schemas import QuestionIntent


COMPARISON_TOPIC_METRICS: dict[str, tuple[str, ...]] = {
    "liquidity": ("current_ratio", "quick_ratio", "operating_cash_flow_ratio"),
    "profitability": (
        "gross_margin", "operating_margin", "net_margin", "roa", "roe",
    ),
    "leverage": ("debt_to_equity", "debt_to_assets", "interest_coverage"),
    "working capital": ("dso", "inventory_days"),
    "costs": ("gross_margin", "operating_margin"),
}


@dataclass(frozen=True)
class RoutedQuestion:
    intent: QuestionIntent
    query_intent: QueryIntent
    comparison: bool
    comparison_metrics: tuple[str, ...]
    include_all_financial_facts: bool
    requires_trend_analysis: bool


class QuestionRouter:
    """Classify common user intents without calculating or inferring values."""

    def __init__(
        self, query_parser: DeterministicQueryParser | None = None
    ) -> None:
        self.query_parser = query_parser or DeterministicQueryParser()

    def route(
        self,
        question: str,
        period: str | None = None,
        *,
        has_comparison_company: bool = False,
    ) -> RoutedQuestion:
        parsed = self.query_parser.parse(question, period)
        normalized = re.sub(r"\s+", " ", question.casefold())
        comparison = has_comparison_company or re.search(
            r"\b(compare|comparison|versus|vs\.?)\b", normalized
        ) is not None
        summary = re.search(
            r"\b(summary|summarize|summarise|overview)\b", normalized
        ) is not None
        why = re.search(
            r"\b(why|explain|reason|cause|commentary|discuss)\b", normalized
        ) is not None

        if comparison:
            intent = QuestionIntent.COMPARISON
        elif parsed.wants_red_flags:
            intent = QuestionIntent.RED_FLAG
        elif why:
            intent = QuestionIntent.WHY
        elif summary:
            intent = QuestionIntent.SUMMARY
        elif parsed.wants_trends:
            intent = QuestionIntent.TREND
        elif parsed.computed_metrics:
            intent = QuestionIntent.RATIO
        else:
            intent = QuestionIntent.EXACT_LOOKUP

        comparison_metrics = list(
            parsed.computed_metrics or parsed.fact_metrics
        )
        if comparison and not comparison_metrics:
            for topic in parsed.topics:
                comparison_metrics.extend(COMPARISON_TOPIC_METRICS.get(topic, ()))
        comparison_metrics = list(dict.fromkeys(comparison_metrics))

        broad_comparison = comparison and not (
            parsed.fact_metrics or parsed.computed_metrics
        )
        requires_trend_analysis = (
            parsed.wants_trends
            or parsed.wants_red_flags
            or (comparison and not (parsed.fact_metrics or parsed.computed_metrics))
        )
        return RoutedQuestion(
            intent=intent,
            query_intent=parsed,
            comparison=comparison,
            comparison_metrics=tuple(comparison_metrics),
            include_all_financial_facts=(
                parsed.wants_red_flags or broad_comparison
            ),
            requires_trend_analysis=requires_trend_analysis,
        )

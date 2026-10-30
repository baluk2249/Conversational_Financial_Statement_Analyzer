"""Narrative Q&A orchestration with strict post-generation validation."""

from __future__ import annotations

import logging
from collections.abc import Callable

from agents.comparison.schemas import ComparisonResult
from agents.narrative_qa.citation_validator import (
    CitationValidator,
    ValidationResult,
)
from agents.narrative_qa.formatters import (
    SafeFallbackFormatter,
    is_causal_question,
    requires_deterministic_refusal,
)
from agents.narrative_qa.prompt_builder import PromptBuilder
from llm.base import LLMClient, LLMProviderError
from models.schemas import AnswerEvidence

logger = logging.getLogger(__name__)


class NarrativeAnswerError(RuntimeError):
    """Raised if neither generated nor deterministic output passes validation."""


class NarrativeSummaryError(RuntimeError):
    """Raised when an evidence-validated LLM summary cannot be produced."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class NarrativeQAAgent:
    """Answer financial questions only from the provided AnswerEvidence."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        prompt_builder: PromptBuilder | None = None,
        citation_validator: CitationValidator | None = None,
        fallback_formatter: SafeFallbackFormatter | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.citation_validator = citation_validator or CitationValidator()
        self.fallback_formatter = fallback_formatter or SafeFallbackFormatter()

    def answer(self, evidence: AnswerEvidence) -> str:
        """Return a validated answer or a deterministic evidence-only fallback."""
        if (
            evidence.status == "INSUFFICIENT_EVIDENCE"
            or requires_deterministic_refusal(evidence)
            or (
                is_causal_question(evidence.question)
                and not any(
                    chunk.company_id == evidence.company_id
                    for chunk in evidence.narrative_evidence
                )
            )
            or self.llm_client is None
        ):
            return self._validated_fallback(evidence)
        system_prompt, user_prompt = self.prompt_builder.build(evidence)
        try:
            draft = self.llm_client.generate(system_prompt, user_prompt)
        except LLMProviderError:
            logger.exception(
                "Narrative provider failed; using deterministic evidence fallback"
            )
            return self._validated_fallback(evidence)
        validation = self.citation_validator.validate(draft, evidence)
        if validation.valid:
            return draft
        logger.warning(
            "Rejected ungrounded narrative answer; retrying once: %s",
            "; ".join(validation.errors),
        )
        retry_system_prompt = (
            system_prompt
            + "\nThe previous draft failed evidence validation for these reasons: "
            + "; ".join(validation.errors)
            + ". Rewrite it from scratch using only supported evidence. Put the "
            "exact source citation immediately after each financial number. "
            "Do not use numbered lists or introduce unsupported figures."
        )
        try:
            draft = self.llm_client.generate(
                retry_system_prompt,
                user_prompt
                + "\n\nProduce a corrected replacement answer. Every financial "
                "number must have its exact supporting citation immediately "
                "after it.",
            )
        except LLMProviderError:
            logger.exception(
                "Narrative provider failed during citation-correction retry; "
                "using deterministic evidence fallback"
            )
            return self._validated_fallback(evidence)
        validation = self.citation_validator.validate(draft, evidence)
        if validation.valid:
            return draft
        logger.warning(
            "Rejected ungrounded narrative answer after correction retry: %s",
            "; ".join(validation.errors),
        )
        return self._validated_fallback(evidence)

    def answer_comparison(
        self,
        result: ComparisonResult,
        *,
        upstream_insufficient: bool = False,
    ) -> str:
        """Explain verified comparison output without weakening refusal rules."""
        if upstream_insufficient or result.status == "INSUFFICIENT_EVIDENCE":
            return self._validated_comparison_fallback(
                result, upstream_insufficient=upstream_insufficient
            )
        if self.llm_client is None:
            return self._validated_comparison_fallback(result)

        system_prompt, user_prompt = self.prompt_builder.build_comparison(result)
        try:
            draft = self.llm_client.generate(system_prompt, user_prompt)
        except LLMProviderError:
            logger.exception(
                "Narrative provider failed; using deterministic comparison fallback"
            )
            return self._validated_comparison_fallback(result)

        validation = self.citation_validator.validate_comparison(draft, result)
        if validation.valid:
            return draft
        logger.warning(
            "Rejected ungrounded comparison answer: %s",
            "; ".join(validation.errors),
        )
        return self._validated_comparison_fallback(result)

    def summarize(
        self, evidence: AnswerEvidence, *, section: str
    ) -> str:
        """Generate a summary only when a configured model passes validation."""
        system_prompt, user_prompt = self.prompt_builder.build_summary(
            evidence, section
        )
        return self._generate_validated_summary(
            system_prompt,
            user_prompt,
            lambda draft: self.citation_validator.validate_summary(
                draft, evidence
            ),
            lambda: self.fallback_formatter.format_summary(evidence, section),
            "narrative",
        )

    def summarize_comparison(self, result: ComparisonResult) -> str:
        """Summarize verified comparison output without weakening its safeguards."""
        system_prompt, user_prompt = (
            self.prompt_builder.build_comparison_summary(result)
        )
        return self._generate_validated_summary(
            system_prompt,
            user_prompt,
            lambda draft: self.citation_validator.validate_comparison_summary(
                draft, result
            ),
            lambda: self.fallback_formatter.format_comparison_summary(result),
            "comparison",
        )

    def _generate_validated_summary(
        self,
        system_prompt: str,
        user_prompt: str,
        validate: Callable[[str], ValidationResult],
        fallback: Callable[[], str],
        summary_type: str,
    ) -> str:
        if self.llm_client is None:
            raise NarrativeSummaryError(
                "LLM summaries are unavailable because no model provider is configured.",
                503,
            )
        try:
            draft = self.llm_client.generate(system_prompt, user_prompt)
            validation = validate(draft)
            if not validation.valid:
                logger.warning(
                    "Rejected ungrounded %s summary: %s",
                    summary_type,
                    "; ".join(validation.errors),
                )
                retry_system_prompt = (
                    system_prompt
                    + "\nThe previous draft failed evidence validation for "
                    "these reasons: "
                    + "; ".join(validation.errors)
                    + ". Rewrite it from scratch. "
                    "Do not include numeric values or "
                    "period years. Remove unsupported claims and mention only "
                    "metric names, evidence-backed statuses, and limitations."
                )
                draft = self.llm_client.generate(
                    retry_system_prompt,
                    user_prompt
                    + "\n\nProduce a corrected replacement summary. Do not "
                    "include any numeric value, period year, or claim that "
                    "cannot be supported by the supplied evidence.",
                )
                validation = validate(draft)
        except LLMProviderError as exc:
            logger.exception("%s summary provider failed", summary_type.title())
            raise NarrativeSummaryError(
                "The configured model provider could not generate a summary.",
                503,
            ) from exc
        if not validation.valid:
            logger.warning(
                "Rejected %s summary after correction attempt; using "
                "evidence-only fallback: %s",
                summary_type,
                "; ".join(validation.errors),
            )
            draft = fallback()
            validation = validate(draft)
            if not validation.valid:
                raise NarrativeSummaryError(
                    "Neither the generated summary nor its evidence-only "
                    "fallback passed validation: "
                    + "; ".join(validation.errors),
                    502,
                )
        return draft.strip()

    def _validated_fallback(self, evidence: AnswerEvidence) -> str:
        answer = self.fallback_formatter.format(evidence)
        validation = self.citation_validator.validate(answer, evidence)
        if not validation.valid:
            raise NarrativeAnswerError(
                "The deterministic fallback did not pass evidence validation: "
                + "; ".join(validation.errors)
            )
        return answer

    def _validated_comparison_fallback(
        self,
        result: ComparisonResult,
        *,
        upstream_insufficient: bool = False,
    ) -> str:
        answer = self.fallback_formatter.format_comparison(
            result, upstream_insufficient=upstream_insufficient
        )
        validation = self.citation_validator.validate_comparison(answer, result)
        if not validation.valid:
            raise NarrativeAnswerError(
                "The deterministic comparison fallback did not pass evidence "
                "validation: " + "; ".join(validation.errors)
            )
        return answer

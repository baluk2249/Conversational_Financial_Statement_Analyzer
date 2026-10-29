"""Narrative Q&A orchestration with strict post-generation validation."""

from __future__ import annotations

import logging

from agents.narrative_qa.citation_validator import CitationValidator
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
            "Rejected ungrounded narrative answer: %s", "; ".join(validation.errors)
        )
        return self._validated_fallback(evidence)

    def _validated_fallback(self, evidence: AnswerEvidence) -> str:
        answer = self.fallback_formatter.format(evidence)
        validation = self.citation_validator.validate(answer, evidence)
        if not validation.valid:
            raise NarrativeAnswerError(
                "The deterministic fallback did not pass evidence validation: "
                + "; ".join(validation.errors)
            )
        return answer

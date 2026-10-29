"""Deterministic value comparison after compatibility validation."""

from __future__ import annotations

from decimal import Decimal

from agents.comparison.schemas import ComparisonEvidence, ComparisonMetric
from agents.comparison.validators import Compatibility


def compare_values(
    metric_name: str,
    evidence_a: ComparisonEvidence | None,
    evidence_b: ComparisonEvidence | None,
    compatibility: Compatibility | None = None,
) -> ComparisonMetric:
    """Compare two values without interpreting which direction is financially better."""
    if evidence_a is None or evidence_b is None:
        return ComparisonMetric(
            metric_name=metric_name,
            company_a=evidence_a,
            company_b=evidence_b,
            comparison_status="INSUFFICIENT_EVIDENCE",
            warning_codes=["MISSING_COMPARISON_EVIDENCE"],
        )

    if compatibility is None:
        raise ValueError("Compatibility validation is required when both values exist")

    warning_codes = list(dict.fromkeys(item.code for item in compatibility.warnings))
    if not compatibility.comparable:
        return ComparisonMetric(
            metric_name=metric_name,
            company_a=evidence_a,
            company_b=evidence_b,
            comparison_status="INCOMPARABLE",
            warning_codes=warning_codes,
        )

    value_a = compatibility.comparison_value_a
    value_b = compatibility.comparison_value_b
    if value_a is None or value_b is None:
        raise ValueError("Comparable values must be present after validation")
    difference = value_b - value_a
    return ComparisonMetric(
        metric_name=metric_name,
        company_a=evidence_a.model_copy(
            update={
                "normalized_value": (
                    value_a if compatibility.normalized_unit is not None else None
                ),
                "normalized_unit": compatibility.normalized_unit,
            }
        ),
        company_b=evidence_b.model_copy(
            update={
                "normalized_value": (
                    value_b if compatibility.normalized_unit is not None else None
                ),
                "normalized_unit": compatibility.normalized_unit,
            }
        ),
        comparison_status="COMPARABLE",
        difference=difference,
        higher_company_id=(
            evidence_b.company_id
            if difference > Decimal("0")
            else evidence_a.company_id if difference < Decimal("0") else None
        ),
        warning_codes=warning_codes,
    )

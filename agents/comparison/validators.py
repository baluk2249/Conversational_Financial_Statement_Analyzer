"""Compatibility checks for values selected from separate evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from agents.comparison.schemas import ComparisonEvidence, ComparisonWarning

UNIT_FACTORS = {
    "absolute": Decimal("1"),
    "unit": Decimal("1"),
    "units": Decimal("1"),
    "thousand": Decimal("1000"),
    "thousands": Decimal("1000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
}
NON_MONETARY_UNITS = {"ratio", "percent", "percentage", "days"}


@dataclass(frozen=True)
class Compatibility:
    comparable: bool
    comparison_value_a: Decimal | None
    comparison_value_b: Decimal | None
    normalized_unit: str | None
    warnings: list[ComparisonWarning]


def validate_comparability(
    a: ComparisonEvidence,
    b: ComparisonEvidence,
    *,
    metric_name: str,
    company_a_id: str,
    company_b_id: str,
    non_monetary: bool,
    same_period_range: bool | None,
    fiscal_year_end_match: bool | None,
) -> Compatibility:
    """Validate metadata and return safe values for deterministic comparison."""
    warnings: list[ComparisonWarning] = []
    blocking = False

    def warn(code: str, message: str, *, blocks: bool = True) -> None:
        nonlocal blocking
        warnings.append(
            ComparisonWarning(
                code=code,
                message=message,
                metric_name=metric_name,
                company_ids=[company_a_id, company_b_id],
            )
        )
        blocking = blocking or blocks

    if a.company_id == b.company_id:
        warn("COMPANY_IDENTITY_MISMATCH", "Both evidence bundles identify the same company.")
    if a.period != b.period:
        warn("FISCAL_PERIOD_MISMATCH", f"Reporting periods differ: {a.period} vs {b.period}.")
    if same_period_range is False:
        warn("FISCAL_PERIOD_MISMATCH", "The reporting-period date ranges differ.")
    elif same_period_range is None:
        warn(
            "MISSING_COMPARISON_EVIDENCE",
            "Reporting-period date ranges could not be verified in both bundles.",
        )
    if fiscal_year_end_match is False:
        warn("FISCAL_YEAR_END_MISMATCH", "Fiscal-year-end dates differ.")
    elif fiscal_year_end_match is None:
        warn(
            "MISSING_COMPARISON_EVIDENCE",
            "Fiscal-year-end metadata could not be verified in both bundles.",
        )

    if (
        not a.metric_definition
        or not b.metric_definition
        or a.metric_definition != b.metric_definition
    ):
        warn("METRIC_DEFINITION_MISMATCH", "Metric definitions do not match.")
    if a.formula_version != b.formula_version:
        if a.formula_version is not None or b.formula_version is not None:
            warn("FORMULA_VERSION_MISMATCH", "Calculated-metric formula versions do not match.")

    if a.currency != b.currency or (
        not non_monetary and (not a.currency or not b.currency)
    ):
        warn(
            "CURRENCY_MISMATCH",
            f"Currencies differ: {a.currency or 'unspecified'} vs {b.currency or 'unspecified'}.",
            blocks=not non_monetary,
        )

    value_a, value_b = a.value, b.value
    normalized_unit: str | None = None
    if non_monetary:
        if a.display_unit != b.display_unit:
            warn(
                "UNIT_MISMATCH",
                f"Calculated metric display units differ: {a.display_unit} vs {b.display_unit}.",
            )
    elif a.unit != b.unit:
        if not a.unit or not b.unit:
            warn("UNIT_MISMATCH", "Monetary unit scale is missing from one or both values.")
        else:
            factor_a = UNIT_FACTORS.get((a.unit or "").strip().lower())
            factor_b = UNIT_FACTORS.get((b.unit or "").strip().lower())
            if factor_a is None or factor_b is None:
                warn(
                    "UNIT_MISMATCH",
                    f"Unit scales cannot be normalized deterministically: {a.unit} vs {b.unit}.",
                )
            else:
                value_a *= factor_a
                value_b *= factor_b
                normalized_unit = "absolute"
                warn(
                    "UNIT_MISMATCH",
                    f"Values were aligned to absolute units from {a.unit} and {b.unit}.",
                    blocks=False,
                )
    elif not non_monetary:
        if not a.unit:
            warn("UNIT_MISMATCH", "Monetary unit scale is missing.")
        normalized_unit = a.unit

    if blocking:
        value_a = value_b = None
    return Compatibility(
        comparable=not blocking,
        comparison_value_a=value_a,
        comparison_value_b=value_b,
        normalized_unit=normalized_unit,
        warnings=warnings,
    )

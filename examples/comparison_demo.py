"""Synthetic, local-only demonstration of validated company comparison."""

from datetime import date
from decimal import Decimal

from agents.comparison import ComparisonAgent, ComparisonRequest
from models.schemas import AnswerEvidence, ComputedMetric, FinancialFact


def _company_evidence(company_id: str, ratio: str, assets: str) -> AnswerEvidence:
    source_fact = FinancialFact(
        fact_id=f"{company_id}-current-assets",
        company_id=company_id,
        company_name=company_id.upper(),
        statement_type="balance_sheet",
        canonical_line_item="current_assets",
        metric_definition="reported_current_assets",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        fiscal_year="FY2025",
        fiscal_year_end=date(2025, 12, 31),
        value=Decimal(assets),
        currency="GBP",
        unit="absolute",
        source_id=f"source-{company_id}",
        source_file=f"{company_id}-annual-report.pdf",
        locator={"page": 18, "label": "Current assets"},
        extraction_confidence=0.99,
    )
    metric = ComputedMetric(
        metric_id=f"{company_id}-current-ratio",
        company_id=company_id,
        metric_name="current_ratio",
        period="FY2025",
        value=Decimal(ratio),
        display_unit="ratio",
        formula_version="current_ratio_v1",
        input_fact_refs=[source_fact.fact_id],
        currency="GBP",
        unit="absolute",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        fiscal_year_end=date(2025, 12, 31),
    )
    return AnswerEvidence(
        question="Compare current ratios.",
        company_id=company_id,
        company_name=company_id.upper(),
        periods=["FY2025"],
        facts=[source_fact],
        computed_metrics=[metric],
    )


if __name__ == "__main__":
    result = ComparisonAgent().compare(
        _company_evidence("abc-ltd", "1.30", "150"),
        _company_evidence("xyz-ltd", "1.85", "185"),
        ComparisonRequest(
            question="Which company has the higher current ratio?",
            metric_names=["current_ratio"],
            period="FY2025",
        ),
    )
    print(result.model_dump_json(indent=2))

"""Run a synthetic grounded financial Q&A example without external APIs."""

from datetime import date
from decimal import Decimal

from agents.narrative_qa import NarrativeQAAgent
from models.schemas import AnswerEvidence, FinancialFact


def main() -> None:
    revenue = FinancialFact(
        fact_id="demo-revenue-fy2025",
        company_id="demo-ltd",
        company_name="Demo Ltd",
        statement_type="income_statement",
        canonical_line_item="revenue",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        fiscal_year="FY2025",
        fiscal_year_end=date(2025, 12, 31),
        value=Decimal("12000000.00"),
        currency="GBP",
        unit="absolute",
        source_id="demo-source",
        source_file="demo-annual-report.pdf",
        locator={"page": 52, "label": "Revenue"},
        extraction_confidence=0.99,
    )
    evidence = AnswerEvidence(
        question="What was Demo Ltd's revenue in FY2025?",
        company_id="demo-ltd",
        company_name="Demo Ltd",
        periods=["FY2025"],
        facts=[revenue],
    )
    print(NarrativeQAAgent().answer(evidence))


if __name__ == "__main__":
    main()

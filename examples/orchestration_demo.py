"""Synthetic, local-only end-to-end orchestration demonstration."""

from datetime import date
from decimal import Decimal

from agents.retrieval_grounding.agent import RetrievalGroundingAgent
from models.schemas import FinancialFact
from orchestration import OrchestrationRequest, Orchestrator
from retrieval.structured_store import SQLiteFinancialStore


def _fact(
    company_id: str,
    metric: str,
    amount: str,
    page: int,
) -> FinancialFact:
    return FinancialFact(
        fact_id=f"synthetic-{company_id}-{metric}-FY2025",
        company_id=company_id,
        company_name=f"Synthetic {company_id.upper()} Ltd",
        statement_type="balance_sheet",
        canonical_line_item=metric,
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        fiscal_year="FY2025",
        fiscal_year_end=date(2025, 12, 31),
        value=Decimal(amount),
        currency="GBP",
        unit="absolute",
        source_id=f"synthetic-source-{company_id}",
        source_file=f"SYNTHETIC_{company_id}_annual_report.pdf",
        locator={"page": page, "label": metric.replace("_", " ").title()},
        extraction_confidence=1.0,
    )


def main() -> None:
    store = SQLiteFinancialStore(":memory:")
    for company, assets, liabilities in (
        ("synthetic-acme", "1500", "1000"),
        ("synthetic-zenith", "1800", "1000"),
    ):
        store.add_fact(_fact(company, "current_assets", assets, 12))
        store.add_fact(_fact(company, "current_liabilities", liabilities, 13))

    orchestrator = Orchestrator(RetrievalGroundingAgent(store))
    for request in (
        OrchestrationRequest(
            question="What was the current ratio in FY2025?",
            company_id="synthetic-acme",
        ),
        OrchestrationRequest(
            question="Compare the current ratio in FY2025.",
            company_id="synthetic-acme",
            comparison_company_id="synthetic-zenith",
        ),
    ):
        response = orchestrator.answer(request)
        print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

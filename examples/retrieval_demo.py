"""Runnable demonstration using clearly synthetic financial statement data."""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal

from agents.retrieval_grounding.agent import RetrievalGroundingAgent
from models.schemas import DocumentChunk, FinancialFact
from retrieval.faiss_store import FaissNarrativeStore
from retrieval.structured_store import SQLiteFinancialStore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    store = SQLiteFinancialStore(".data/synthetic-financial-facts.sqlite3")
    synthetic_facts = [
        FinancialFact(
            fact_id="synthetic-current-assets-fy2025",
            company_id="synthetic-acme",
            company_name="Synthetic Acme Ltd",
            statement_type="balance_sheet",
            canonical_line_item="current_assets",
            period_start=date(2025, 1, 1), period_end=date(2025, 12, 31),
            fiscal_year="FY2025", fiscal_year_end=date(2025, 12, 31),
            value=Decimal("1500000"), currency="GBP", unit="absolute",
            source_id="synthetic-source-fy2025", source_file="SYNTHETIC_annual_report.pdf",
            locator={"page": 21, "label": "Current assets"}, extraction_confidence=1.0,
        ),
        FinancialFact(
            fact_id="synthetic-current-liabilities-fy2025",
            company_id="synthetic-acme",
            company_name="Synthetic Acme Ltd",
            statement_type="balance_sheet",
            canonical_line_item="current_liabilities",
            period_start=date(2025, 1, 1), period_end=date(2025, 12, 31),
            fiscal_year="FY2025", fiscal_year_end=date(2025, 12, 31),
            value=Decimal("1000000"), currency="GBP", unit="absolute",
            source_id="synthetic-source-fy2025", source_file="SYNTHETIC_annual_report.pdf",
            locator={"page": 21, "label": "Current liabilities"}, extraction_confidence=1.0,
        ),
    ]
    for fact in synthetic_facts:
        store.add_fact(fact)

    narrative_store = FaissNarrativeStore(persist_directory=".data/faiss")
    narrative_store.add_chunks([
        DocumentChunk(
            chunk_id="synthetic-working-capital-fy2025",
            company_id="synthetic-acme",
            company_name="Synthetic Acme Ltd",
            document_type="annual_report",
            statement_type="management_commentary",
            section="Working capital",
            period="FY2025",
            source_id="synthetic-source-fy2025",
            source_file="SYNTHETIC_annual_report.pdf",
            page=38,
            text="SYNTHETIC TEST DATA: management commentary describes stable supplier payment terms.",
        )
    ])

    agent = RetrievalGroundingAgent(store, narrative_store)
    result = agent.retrieve_context("What was the current ratio in FY2025?", "synthetic-acme")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
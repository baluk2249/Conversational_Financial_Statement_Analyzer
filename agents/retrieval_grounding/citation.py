"""Human-readable citations derived only from source metadata."""

from models.schemas import FinancialFact, SourceReference


def build_fact_citation(fact: FinancialFact) -> str:
    """Format the shared PDF or spreadsheet citation representation."""
    return fact.source_reference.citation


def build_source_citation(source: SourceReference) -> str:
    return source.citation
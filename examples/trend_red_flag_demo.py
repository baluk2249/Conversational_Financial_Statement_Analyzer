"""Synthetic, local-only example of deterministic trend and flag analysis."""

from datetime import date
from decimal import Decimal

from agents.trend_red_flag.agent import TrendRedFlagAgent
from models.schemas import FinancialFact


def fact(metric: str, period: str, year: int, value: str, statement: str) -> FinancialFact:
    return FinancialFact(
        fact_id=f"demo-{metric}-{period}",
        company_id="demo-co",
        company_name="Demo Company",
        statement_type=statement,
        canonical_line_item=metric,
        period_start=date(year, 1, 1),
        period_end=date(year, 12, 31),
        fiscal_year=period,
        fiscal_year_end=date(year, 12, 31),
        value=Decimal(value),
        currency="GBP",
        unit="absolute",
        source_id=f"demo-source-{period}",
        source_file=f"synthetic-{period}.pdf",
        locator={"page": 1, "label": metric.replace("_", " ").title()},
        extraction_confidence=1.0,
    )


facts = [
    fact("revenue", "FY2024", 2024, "1000", "income_statement"),
    fact("operating_profit", "FY2024", 2024, "150", "income_statement"),
    fact("net_income", "FY2024", 2024, "100", "income_statement"),
    fact("current_assets", "FY2024", 2024, "500", "balance_sheet"),
    fact("current_liabilities", "FY2024", 2024, "250", "balance_sheet"),
    fact("inventory", "FY2024", 2024, "80", "balance_sheet"),
    fact("total_debt", "FY2024", 2024, "300", "balance_sheet"),
    fact("equity", "FY2024", 2024, "400", "balance_sheet"),
    fact("operating_cash_flow", "FY2024", 2024, "140", "cash_flow"),
    fact("revenue", "FY2025", 2025, "1100", "income_statement"),
    fact("operating_profit", "FY2025", 2025, "110", "income_statement"),
    fact("net_income", "FY2025", 2025, "70", "income_statement"),
    fact("current_assets", "FY2025", 2025, "520", "balance_sheet"),
    fact("current_liabilities", "FY2025", 2025, "300", "balance_sheet"),
    fact("inventory", "FY2025", 2025, "120", "balance_sheet"),
    fact("total_debt", "FY2025", 2025, "350", "balance_sheet"),
    fact("equity", "FY2025", 2025, "420", "balance_sheet"),
    fact("operating_cash_flow", "FY2025", 2025, "30", "cash_flow"),
]

analysis = TrendRedFlagAgent().analyse("demo-co", facts, ["FY2024", "FY2025"])
print("Trends:")
for trend in analysis.trends:
    if trend.quality_status == "VALID" and trend.delta:
        print(f"  {trend.metric_name}: {trend.from_period} -> {trend.to_period}: {trend.delta}")
print("Red flags:")
for flag in analysis.red_flags:
    print(f"  {flag.flag_type}: {flag.status} ({flag.explanation_code})")

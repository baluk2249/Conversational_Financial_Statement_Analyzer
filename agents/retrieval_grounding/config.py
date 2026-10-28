"""MVP aliases and formula input mappings; safe to extend without agent changes."""

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "sales", "turnover"),
    "gross_profit": ("gross profit",),
    "operating_profit": ("operating profit", "operating income", "ebit"),
    "net_income": ("net income", "profit after tax", "net profit", "profit for the year"),
    "ebitda": ("ebitda",),
    "current_assets": ("current assets",),
    "current_liabilities": ("current liabilities",),
    "inventory": ("inventory", "stock"),
    "accounts_receivable": ("accounts receivable", "receivables", "debtors", "trade receivables"),
    "total_debt": ("total debt", "debt", "borrowings"),
    "equity": ("equity", "shareholders' equity", "shareholders equity"),
    "total_assets": ("total assets",),
    "operating_cash_flow": ("operating cash flow", "cash flow from operations"),
    "interest_expense": ("interest expense", "finance costs"),
    "cost_of_goods_sold": ("cost of goods sold", "cost of sales"),
    "ebit": ("ebit", "earnings before interest and tax"),
}

COMPUTED_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "current_ratio": ("current ratio",),
    "quick_ratio": ("quick ratio", "acid test", "acid-test ratio"),
    "operating_cash_flow_ratio": ("operating cash flow ratio", "cash flow ratio"),
    "debt_to_equity": ("debt to equity", "debt-to-equity", "gearing ratio"),
    "debt_to_assets": ("debt to assets", "debt-to-assets"),
    "interest_coverage": ("interest coverage", "times interest earned"),
    "gross_margin": ("gross margin",),
    "operating_margin": ("operating margin",),
    "net_margin": ("net margin",),
    "roa": ("return on assets", "roa"),
    "roe": ("return on equity", "roe"),
    "dso": ("dso", "days sales outstanding", "days' sales outstanding"),
    "inventory_days": ("inventory days", "days inventory outstanding"),
    "asset_turnover": ("asset turnover",),
}

COMPUTED_METRIC_INPUTS: dict[str, tuple[str, ...]] = {
    "current_ratio": ("current_assets", "current_liabilities"),
    "quick_ratio": ("current_assets", "inventory", "current_liabilities"),
    "operating_cash_flow_ratio": ("operating_cash_flow", "current_liabilities"),
    "debt_to_equity": ("total_debt", "equity"),
    "debt_to_assets": ("total_debt", "total_assets"),
    "interest_coverage": ("ebit", "interest_expense"),
    "gross_margin": ("gross_profit", "revenue"),
    "operating_margin": ("operating_profit", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "roa": ("net_income", "total_assets"),
    "roe": ("net_income", "equity"),
    "dso": ("accounts_receivable", "revenue"),
    "inventory_days": ("inventory", "cost_of_goods_sold"),
    "asset_turnover": ("revenue", "total_assets"),
}

WARNING_CODES = {
    "CURRENCY_MISMATCH",
    "FISCAL_PERIOD_MISMATCH",
    "FISCAL_YEAR_END_MISMATCH",
    "UNIT_MISMATCH",
    "METRIC_DEFINITION_MISMATCH",
}
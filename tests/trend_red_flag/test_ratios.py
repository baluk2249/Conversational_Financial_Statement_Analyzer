from decimal import Decimal

from finance.ratios import calculate_ratio
from models.schemas import SourceLocator


def test_current_and_quick_ratio_are_decimal_and_versioned(fact_factory):
    facts = [
        fact_factory("current_assets", "FY2025", "150"),
        fact_factory("current_liabilities", "FY2025", "100"),
        fact_factory("inventory", "FY2025", "25"),
    ]
    current = calculate_ratio(facts, "abc", "current_ratio", "FY2025")
    quick = calculate_ratio(facts, "abc", "quick_ratio", "FY2025")
    assert current.value == Decimal("1.5000")
    assert quick.value == Decimal("1.2500")
    assert current.formula_version == "current_ratio_v1"
    assert current.quality_status == "VALID"
    assert set(quick.input_fact_refs) == {fact.fact_id for fact in facts}


def test_identical_duplicate_facts_are_used_with_all_provenance(fact_factory):
    assets = fact_factory("current_assets", "FY2025", "150")
    liabilities = fact_factory("current_liabilities", "FY2025", "100")
    duplicate_assets = assets.model_copy(
        update={
            "fact_id": "duplicate-assets",
            "source_id": "duplicate-source",
            "source_file": "second-upload.pdf",
        }
    )
    duplicate_liabilities = liabilities.model_copy(
        update={
            "fact_id": "duplicate-liabilities",
            "source_id": "duplicate-source",
            "source_file": "second-upload.pdf",
        }
    )

    result = calculate_ratio(
        [assets, duplicate_assets, liabilities, duplicate_liabilities],
        "abc",
        "current_ratio",
        "FY2025",
    )

    assert result.value == Decimal("1.5000")
    assert result.quality_status == "VALID"
    assert set(result.input_fact_refs) == {
        assets.fact_id,
        duplicate_assets.fact_id,
        liabilities.fact_id,
        duplicate_liabilities.fact_id,
    }


def test_conflicting_duplicate_fact_is_ambiguous_not_missing(fact_factory):
    assets = fact_factory("current_assets", "FY2025", "150")
    conflicting_assets = assets.model_copy(
        update={"fact_id": "conflicting-assets", "value": Decimal("160")}
    )
    liabilities = fact_factory("current_liabilities", "FY2025", "100")

    result = calculate_ratio(
        [assets, conflicting_assets, liabilities],
        "abc",
        "current_ratio",
        "FY2025",
    )

    assert result.value is None
    assert result.quality_status == "INCOMPATIBLE_INPUTS"
    assert "AMBIGUOUS_INPUT:current_assets" in result.warnings
    assert "MISSING_INPUT:current_assets" not in result.warnings


def test_debt_to_equity_and_margins(fact_factory):
    facts = [
        fact_factory("total_debt", "FY2025", "80"),
        fact_factory("equity", "FY2025", "100"),
        fact_factory("operating_profit", "FY2025", "15"),
        fact_factory("net_income", "FY2025", "10"),
        fact_factory("revenue", "FY2025", "100"),
    ]
    assert calculate_ratio(facts, "abc", "debt_to_equity", "FY2025").value == Decimal("0.8000")
    assert calculate_ratio(facts, "abc", "operating_margin", "FY2025").value == Decimal("0.1500")
    assert calculate_ratio(facts, "abc", "net_margin", "FY2025").value == Decimal("0.1000")


def test_dso_and_inventory_days_need_adjacent_balance_sheet_facts(fact_factory):
    facts = [
        fact_factory("accounts_receivable", "FY2024", "10"),
        fact_factory("accounts_receivable", "FY2025", "20"),
        fact_factory("inventory", "FY2024", "20"),
        fact_factory("inventory", "FY2025", "30"),
        fact_factory("revenue", "FY2025", "100"),
        fact_factory("cost_of_goods_sold", "FY2025", "-100"),
    ]
    dso = calculate_ratio(facts, "abc", "dso", "FY2025")
    inventory_days = calculate_ratio(facts, "abc", "inventory_days", "FY2025")
    assert dso.value == Decimal("54.7500")
    assert inventory_days.value == Decimal("91.2500")
    assert set(dso.input_fact_refs) == {
        "abc-accounts_receivable-FY2024",
        "abc-accounts_receivable-FY2025",
        "abc-revenue-FY2025",
    }


def test_missing_inputs_and_zero_denominator_are_unavailable_not_zero(fact_factory):
    missing = calculate_ratio(
        [fact_factory("current_assets", "FY2025", "100")],
        "abc",
        "current_ratio",
        "FY2025",
    )
    divide_by_zero = calculate_ratio(
        [
            fact_factory("current_assets", "FY2025", "100"),
            fact_factory("current_liabilities", "FY2025", "0"),
        ],
        "abc",
        "current_ratio",
        "FY2025",
    )
    assert missing.value is None
    assert missing.quality_status == "INSUFFICIENT_EVIDENCE"
    assert "MISSING_INPUT:current_liabilities" in missing.warnings
    assert missing.input_fact_refs == ["abc-current_assets-FY2025"]
    assert divide_by_zero.value is None
    assert divide_by_zero.quality_status == "INSUFFICIENT_EVIDENCE"
    assert "MISSING_INPUT:non_zero_denominator" in divide_by_zero.warnings


def test_days_based_ratio_does_not_assume_365_days(fact_factory):
    revenue = fact_factory("revenue", "FY2025", "100").model_copy(
        update={"period_start": None, "period_end": None}
    )
    facts = [
        fact_factory("accounts_receivable", "FY2024", "10"),
        fact_factory("accounts_receivable", "FY2025", "20").model_copy(
            update={"period_start": None}
        ),
        revenue,
    ]
    result = calculate_ratio(facts, "abc", "dso", "FY2025")
    assert result.value is None
    assert "MISSING_INPUT:period_start_and_end" in result.warnings


def test_invalid_period_range_is_incompatible_not_a_negative_duration(fact_factory):
    ordinary_revenue = fact_factory("revenue", "FY2025", "100")
    revenue = ordinary_revenue.model_copy(
        update={
            "period_start": ordinary_revenue.period_end,
            "period_end": ordinary_revenue.period_end.replace(day=1),
        }
    )
    facts = [
        fact_factory("accounts_receivable", "FY2024", "10"),
        fact_factory("accounts_receivable", "FY2025", "20"),
        revenue,
    ]
    result = calculate_ratio(facts, "abc", "dso", "FY2025")
    assert result.value is None
    assert result.quality_status == "INCOMPATIBLE_INPUTS"
    assert "INVALID_PERIOD_RANGE" in result.warnings


def test_metric_formula_records_all_sources_for_average_balance(fact_factory):
    facts = [
        fact_factory("total_assets", "FY2024", "80"),
        fact_factory("total_assets", "FY2025", "120"),
        fact_factory("net_income", "FY2025", "20"),
    ]
    result = calculate_ratio(facts, "abc", "roa", "FY2025")
    assert result.value == Decimal("0.2000")
    assert result.input_fact_refs == [
        "abc-net_income-FY2025",
        "abc-total_assets-FY2025",
        "abc-total_assets-FY2024",
    ]
    assert result.formula_version == "roa_v1"


def test_adjacent_fiscal_years_support_average_balance_ratios_without_dates(
    fact_factory,
):
    facts = [
        fact_factory("net_income", "FY2025", "20"),
        fact_factory("revenue", "FY2025", "100"),
        fact_factory("total_assets", "FY2024", "80"),
        fact_factory("total_assets", "FY2025", "100"),
        fact_factory("equity", "FY2024", "40"),
        fact_factory("equity", "FY2025", "60"),
    ]
    facts = [
        fact.model_copy(
            update={
                "period_start": None,
                "period_end": None,
                "fiscal_year_end": None,
            }
        )
        for fact in facts
    ]

    roa = calculate_ratio(facts, "abc", "roa", "FY2025")
    roe = calculate_ratio(facts, "abc", "roe", "FY2025")
    asset_turnover = calculate_ratio(facts, "abc", "asset_turnover", "FY2025")

    assert roa.value == Decimal("0.2222")
    assert roe.value == Decimal("0.4000")
    assert asset_turnover.value == Decimal("1.1111")
    assert roa.input_fact_refs[-1] == "abc-total_assets-FY2024"


def test_interest_coverage_uses_reported_operating_profit_as_ebit(fact_factory):
    facts = [
        fact_factory("operating_profit", "FY2025", "24"),
        fact_factory("interest_expense", "FY2025", "-3"),
    ]

    result = calculate_ratio(facts, "abc", "interest_coverage", "FY2025")

    assert result.value == Decimal("8.0000")
    assert result.quality_status == "VALID"
    assert result.formula_version == (
        "interest_coverage_v2+operating_profit_as_ebit_v1"
    )
    assert set(result.input_fact_refs) == {fact.fact_id for fact in facts}


def test_debt_ratios_sum_reported_borrowing_components(fact_factory):
    long_term = fact_factory(
        "debt_component", "FY2025", "80"
    ).model_copy(
        update={
            "locator": SourceLocator(
                page=2,
                label="Borrowings",
                details={"table_index": 1, "table_row": 10},
            )
        }
    )
    current = fact_factory(
        "debt_component", "FY2025", "20"
    ).model_copy(
        update={
            "fact_id": "current-borrowings",
            "locator": SourceLocator(
                page=2,
                label="Borrowings",
                details={"table_index": 1, "table_row": 15},
            ),
        }
    )
    equity = fact_factory("equity", "FY2025", "50")

    result = calculate_ratio(
        [long_term, current, equity], "abc", "debt_to_equity", "FY2025"
    )

    assert result.value == Decimal("2.0000")
    assert result.formula_version == (
        "debt_to_equity_v1+total_debt_components_v1"
    )
    assert set(result.input_fact_refs) == {
        long_term.fact_id,
        current.fact_id,
        equity.fact_id,
    }


def test_interest_coverage_derives_ebit_from_profit_before_tax(fact_factory):
    facts = [
        fact_factory("profit_before_tax", "FY2025", "90"),
        fact_factory("interest_expense", "FY2025", "-10"),
    ]

    result = calculate_ratio(facts, "abc", "interest_coverage", "FY2025")

    assert result.value == Decimal("10.0000")
    assert result.formula_version == (
        "interest_coverage_v2+ebit_from_profit_before_tax_v2"
    )
    assert set(result.input_fact_refs) == {
        "abc-profit_before_tax-FY2025",
        "abc-interest_expense-FY2025",
    }

from decimal import Decimal

from finance.ratios import calculate_ratio


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
        fact_factory("cost_of_goods_sold", "FY2025", "100"),
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

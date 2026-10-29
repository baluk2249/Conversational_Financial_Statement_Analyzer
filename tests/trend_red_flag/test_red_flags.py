from decimal import Decimal

from agents.trend_red_flag.red_flags import evaluate_red_flags
from agents.trend_red_flag.rules import load_rules
from agents.trend_red_flag.trends import analyse_trends


def _flags(company_id, periods, facts, metrics, rules=None):
    trends = [
        trend
        for metric_name in (
            "dso", "gross_margin", "operating_margin", "net_margin",
            "debt_to_equity", "debt_to_assets", "interest_coverage",
            "current_ratio", "quick_ratio", "operating_cash_flow_ratio",
        )
        for trend in analyse_trends(company_id, metric_name, periods, metrics)
    ]
    return evaluate_red_flags(
        company_id, periods, facts, metrics, trends, rules or load_rules()
    )


def test_margin_compression_and_threshold_boundary(metric_factory, fact_factory):
    facts = [
        fact_factory("net_income", "FY2025", "10"),
        fact_factory("operating_cash_flow", "FY2025", "10", statement="cash_flow"),
    ]
    metrics = [
        metric_factory("gross_margin", "FY2024", "0.3"),
        metric_factory("gross_margin", "FY2025", "0.29"),
        metric_factory("operating_margin", "FY2024", "0.2"),
        metric_factory("operating_margin", "FY2025", "0.2"),
    ]
    flags = {flag.flag_type: flag for flag in _flags("abc", ["FY2024", "FY2025"], facts, metrics)}
    assert flags["margin_compression"].status == "TRIGGERED"
    assert flags["margin_compression"].rule_version == "margin_compression_v1"
    assert "abc-gross_margin-FY2025" in flags["margin_compression"].supporting_refs

    metrics[1] = metrics[0].model_copy(update={"period": "FY2025"})
    metrics[0] = metrics[0].model_copy(update={"value": Decimal("0.3")})
    boundary_flags = {
        flag.flag_type: flag
        for flag in _flags("abc", ["FY2024", "FY2025"], facts, metrics)
    }
    assert boundary_flags["margin_compression"].status == "NOT_TRIGGERED"


def test_dso_climbing_leverage_and_liquidity_flags(metric_factory, fact_factory):
    facts = [
        fact_factory("net_income", "FY2025", "10"),
        fact_factory("operating_cash_flow", "FY2025", "10", statement="cash_flow"),
    ]
    names_and_values = {
        "dso": ("40", "45"),
        "debt_to_equity": ("0.5", "0.7"),
        "interest_coverage": ("5", "4"),
        "current_ratio": ("2", "1.5"),
        "quick_ratio": ("1.5", "1.2"),
    }
    metrics = [
        metric_factory(name, period, value)
        for name, pair in names_and_values.items()
        for period, value in zip(("FY2024", "FY2025"), pair)
    ]
    flags = {flag.flag_type: flag for flag in _flags("abc", ["FY2024", "FY2025"], facts, metrics)}
    assert flags["dso_climbing"].status == "TRIGGERED"
    assert flags["leverage_stress"].status == "TRIGGERED"
    assert flags["liquidity_deterioration"].status == "TRIGGERED"
    assert flags["leverage_stress"].supporting_refs


def test_inventory_build_up_compares_inventory_growth_to_revenue(fact_factory):
    facts = [
        fact_factory("inventory", "FY2024", "100", statement="balance_sheet"),
        fact_factory("inventory", "FY2025", "130", statement="balance_sheet"),
        fact_factory("revenue", "FY2024", "100"),
        fact_factory("revenue", "FY2025", "110"),
        fact_factory("net_income", "FY2025", "10"),
        fact_factory("operating_cash_flow", "FY2025", "10", statement="cash_flow"),
    ]
    flags = {flag.flag_type: flag for flag in _flags("abc", ["FY2024", "FY2025"], facts, [])}
    inventory = flags["inventory_build_up"]
    assert inventory.status == "TRIGGERED"
    assert inventory.explanation_code == "INVENTORY_GROWTH_EXCEEDS_REVENUE"
    assert {fact_id for fact_id in inventory.supporting_refs} == {
        "abc-inventory-FY2024",
        "abc-inventory-FY2025",
        "abc-revenue-FY2024",
        "abc-revenue-FY2025",
    }


def test_accrual_quality_gap_requires_both_evidence_facts(fact_factory):
    facts = [
        fact_factory("net_income", "FY2025", "100"),
        fact_factory("operating_cash_flow", "FY2025", "20", statement="cash_flow"),
    ]
    flags = {flag.flag_type: flag for flag in _flags("abc", ["FY2025"], facts, [])}
    gap = flags["accrual_quality_gap"]
    assert gap.status == "TRIGGERED"
    assert gap.rule_version == "accrual_quality_gap_v1"
    assert set(gap.supporting_refs) == {fact.fact_id for fact in facts}

    missing = {flag.flag_type: flag for flag in _flags("abc", ["FY2025"], facts[:1], [])}
    assert missing["accrual_quality_gap"].status == "INSUFFICIENT_EVIDENCE"


def test_threshold_missing_and_single_period_are_explicit(metric_factory):
    config = load_rules().definitions.copy()
    del config["margin_compression"]
    from agents.trend_red_flag.rules import RuleDefinition, RuleSet

    config["margin_compression"] = RuleDefinition(
        "margin_compression_custom_v1", True, {}
    )
    rules = RuleSet(config)
    metrics = [
        metric_factory("gross_margin", "FY2024", "0.3"),
        metric_factory("gross_margin", "FY2025", "0.2"),
    ]
    from agents.trend_red_flag.trends import analyse_trends

    trends = analyse_trends("abc", "gross_margin", ["FY2024", "FY2025"], metrics)
    result = evaluate_red_flags(
        "abc", ["FY2024", "FY2025"], [], metrics, trends, rules
    )
    by_type = {flag.flag_type: flag for flag in result}
    assert by_type["margin_compression"].status == "INSUFFICIENT_EVIDENCE"
    assert "THRESHOLD_NOT_CONFIGURED" == by_type["margin_compression"].explanation_code

    single = evaluate_red_flags("abc", ["FY2025"], [], [], [], load_rules())
    assert {flag.status for flag in single if flag.flag_type != "accrual_quality_gap"} == {
        "INSUFFICIENT_EVIDENCE"
    }


def test_each_detector_uses_configured_threshold_boundaries(metric_factory, fact_factory):
    from agents.trend_red_flag.rules import RuleDefinition, RuleSet

    definitions = load_rules().definitions.copy()
    thresholds = {
        "accrual_quality_gap": ("minimum_gap_ratio", "0.5"),
        "dso_climbing": ("minimum_increase", "2"),
        "margin_compression": ("minimum_decline", "0.05"),
        "leverage_stress": ("minimum_worsening", "0.1"),
        "inventory_build_up": ("minimum_growth_gap", "0.1"),
        "liquidity_deterioration": ("minimum_decline", "0.2"),
    }
    for flag_type, (parameter, value) in thresholds.items():
        definitions[flag_type] = RuleDefinition(
            definitions[flag_type].rule_version,
            True,
            {parameter: Decimal(value)},
        )
    rules = RuleSet(definitions)

    def boundary_facts(cash_flow: str, inventory: str) -> list:
        return [
            fact_factory("net_income", "FY2025", "100"),
            fact_factory("operating_cash_flow", "FY2025", cash_flow, statement="cash_flow"),
            fact_factory("inventory", "FY2024", "100", statement="balance_sheet"),
            fact_factory("inventory", "FY2025", inventory, statement="balance_sheet"),
            fact_factory("revenue", "FY2024", "100"),
            fact_factory("revenue", "FY2025", "110"),
        ]

    def boundary_metrics(
        dso_after: str = "12",
        margin_after: str = "0.15",
        debt_after: str = "0.6",
        current_ratio_after: str = "1.8",
    ) -> list:
        return [
            metric_factory("dso", "FY2024", "10"),
            metric_factory("dso", "FY2025", dso_after),
            metric_factory("operating_margin", "FY2024", "0.2"),
            metric_factory("operating_margin", "FY2025", margin_after),
            metric_factory("debt_to_equity", "FY2024", "0.5"),
            metric_factory("debt_to_equity", "FY2025", debt_after),
            metric_factory("current_ratio", "FY2024", "2"),
            metric_factory("current_ratio", "FY2025", current_ratio_after),
        ]

    exact_flags = {
        flag.flag_type: flag
        for flag in _flags(
            "abc",
            ["FY2024", "FY2025"],
            boundary_facts("50", "120"),
            boundary_metrics(),
            rules,
        )
    }
    assert {
        flag_type: exact_flags[flag_type].status
        for flag_type in thresholds
    } == {
        "accrual_quality_gap": "NOT_TRIGGERED",
        "dso_climbing": "NOT_TRIGGERED",
        "margin_compression": "NOT_TRIGGERED",
        "leverage_stress": "NOT_TRIGGERED",
        "inventory_build_up": "NOT_TRIGGERED",
        "liquidity_deterioration": "NOT_TRIGGERED",
    }

    beyond_flags = {
        flag.flag_type: flag
        for flag in _flags(
            "abc",
            ["FY2024", "FY2025"],
            boundary_facts("49", "121"),
            boundary_metrics("12.0001", "0.1499", "0.6001", "1.7999"),
            rules,
        )
    }
    assert {
        flag_type: beyond_flags[flag_type].status
        for flag_type in thresholds
    } == {flag_type: "TRIGGERED" for flag_type in thresholds}

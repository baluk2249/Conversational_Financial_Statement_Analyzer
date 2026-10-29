"""Load and validate version-controlled red-flag rule configuration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

RULE_TYPES = (
    "accrual_quality_gap",
    "dso_climbing",
    "margin_compression",
    "leverage_stress",
    "inventory_build_up",
    "liquidity_deterioration",
)
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "red_flag_rules.yaml"


@dataclass(frozen=True)
class RuleDefinition:
    rule_version: str
    enabled: bool
    parameters: dict[str, Decimal]


@dataclass(frozen=True)
class RuleSet:
    definitions: dict[str, RuleDefinition]

    def get(self, flag_type: str) -> RuleDefinition | None:
        return self.definitions.get(flag_type)


def load_rules(path: str | Path | None = None) -> RuleSet:
    """Load YAML rules; invalid or missing configuration fails explicitly."""
    config_path = Path(path) if path is not None else DEFAULT_RULES_PATH
    with config_path.open("r", encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"Red-flag rules must be a YAML mapping: {config_path}")

    definitions: dict[str, RuleDefinition] = {}
    for flag_type, entry in raw.items():
        if flag_type not in RULE_TYPES:
            raise ValueError(f"Unknown red-flag rule: {flag_type}")
        if not isinstance(entry, dict):
            raise ValueError(f"Rule {flag_type} must be a YAML mapping")
        version = entry.get("rule_version")
        enabled = entry.get("enabled")
        parameters = entry.get("parameters", {})
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Rule {flag_type} requires a rule_version")
        if not isinstance(enabled, bool):
            raise ValueError(f"Rule {flag_type} requires a boolean enabled value")
        if not isinstance(parameters, dict):
            raise ValueError(f"Rule {flag_type} parameters must be a mapping")
        parsed_parameters: dict[str, Decimal] = {}
        for name, value in parameters.items():
            try:
                parsed_parameters[name] = Decimal(str(value))
            except (InvalidOperation, ValueError) as error:
                raise ValueError(
                    f"Rule {flag_type} parameter {name} must be a decimal"
                ) from error
        definitions[flag_type] = RuleDefinition(version, enabled, parsed_parameters)

    return RuleSet(definitions)

"""Deterministic trend and red-flag analysis."""

from agents.trend_red_flag.agent import TrendRedFlagAgent, TrendRedFlagAnalysis
from agents.trend_red_flag.rules import RuleSet, load_rules
from agents.trend_red_flag.trends import analyse_trend, analyse_trends

__all__ = [
    "RuleSet",
    "TrendRedFlagAgent",
    "TrendRedFlagAnalysis",
    "analyse_trend",
    "analyse_trends",
    "load_rules",
]

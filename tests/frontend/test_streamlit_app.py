from frontend import streamlit_app
from frontend.api_client import AnalyzerAPIError
from frontend.streamlit_app import (
    _comparison_value,
    _company_display_labels,
    _summary_action,
)


def test_comparison_value_formats_percentages_and_keeps_citations() -> None:
    value = _comparison_value(
        {
            "value": "0.3776",
            "display_unit": "percent",
            "citations": ["[Income Statement | FY2025 | Gross Profit | p.1]"],
        }
    )

    assert value == (
        "37.76% ([Income Statement | FY2025 | Gross Profit | p.1])"
    )


def test_comparison_value_keeps_ratio_units() -> None:
    assert _comparison_value(
        {"value": "1.2222", "display_unit": "ratio", "citations": []}
    ) == "1.2222 ratio"


def test_duplicate_company_names_are_disambiguated_with_company_ids() -> None:
    labels = _company_display_labels(
        ["cedar-old", "cedar-new", "northstar"],
        {
            "cedar-old": "Cedar Demo PLC",
            "cedar-new": "CEDAR DEMO PLC",
            "northstar": "Northstar Demo Ltd",
        },
    )

    assert labels == {
        "cedar-old": "Cedar Demo PLC (cedar-old)",
        "cedar-new": "CEDAR DEMO PLC (cedar-new)",
        "northstar": "Northstar Demo Ltd",
    }


def test_summary_action_persists_generated_summary(monkeypatch) -> None:
    session_state: dict[str, str] = {}
    shown: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "session_state", session_state)
    monkeypatch.setattr(streamlit_app.st, "button", lambda *args, **kwargs: True)
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda value: shown.append(value))
    monkeypatch.setattr(streamlit_app.st, "info", lambda value: shown.append(value))

    _summary_action("ratio-summary:acme:latest:Liquidity", lambda: {
        "summary": "Liquidity summary."
    })

    assert session_state[
        "ratio-summary:acme:latest:Liquidity:value"
    ] == "Liquidity summary."
    assert shown == ["**AI summary**", "Liquidity summary."]


def test_summary_action_clears_stale_summary_after_api_error(monkeypatch) -> None:
    key = "ratio-summary:acme:latest:Liquidity"
    session_state = {f"{key}:value": "Outdated summary."}
    errors: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "session_state", session_state)
    monkeypatch.setattr(streamlit_app.st, "button", lambda *args, **kwargs: True)
    monkeypatch.setattr(streamlit_app.st, "error", lambda value: errors.append(value))
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda _value: None)
    monkeypatch.setattr(streamlit_app.st, "info", lambda _value: None)

    def fail() -> dict[str, str]:
        raise AnalyzerAPIError("The generated summary did not pass validation.", 502)

    _summary_action(key, fail)

    assert f"{key}:value" not in session_state
    assert errors == ["The generated summary did not pass validation."]

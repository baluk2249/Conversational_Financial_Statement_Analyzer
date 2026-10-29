"""Streamlit shell; all financial operations are delegated to FastAPI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.api_client import AnalyzerAPIError, FinancialAnalyzerClient

st.set_page_config(
    page_title="Conversational Financial Statement Analyzer",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    st.title("Conversational Financial Statement Analyzer")
    client = FinancialAnalyzerClient()
    try:
        health = client.health()
        if health.get("status") != "healthy":
            st.warning("The analyzer is running with limited backend availability.")
    except AnalyzerAPIError as exc:
        st.error(str(exc))
        return

    with st.sidebar:
        st.header("Financial statements")
        upload = st.file_uploader(
            "Upload a PDF, XLSX, or CSV",
            type=["pdf", "xlsx", "csv"],
        )
        upload_company_name = st.text_input(
            "Company name (required for PDF)"
        )
        upload_company_id = st.text_input("Company ID (optional)")
        upload_period = st.text_input("Fiscal period (optional, e.g. FY2025)")
        if upload is not None and st.button("Process document", type="primary"):
            with st.spinner("Sending document to the analyzer..."):
                try:
                    processed = client.upload_document(
                        upload.name,
                        upload.getvalue(),
                        company_name=upload_company_name or None,
                        company_id=upload_company_id or None,
                        period=upload_period or None,
                    )
                    st.success("Document processed by the backend.")
                    st.json(processed)
                except AnalyzerAPIError as exc:
                    st.error(str(exc))

    try:
        companies = client.list_companies()
    except AnalyzerAPIError as exc:
        st.error(str(exc))
        return
    if not companies:
        st.info("Upload a supported statement to begin. PDF uploads add narrative evidence; normalized CSV/XLSX rows add exact financial facts.")
        return

    company_names = {
        item["company_id"]: item["company_name"] for item in companies
    }
    company_ids = list(company_names)
    with st.sidebar:
        company_id = st.selectbox(
            "Company",
            company_ids,
            format_func=lambda value: company_names[value],
        )
        try:
            periods = client.get_periods(company_id)
        except AnalyzerAPIError as exc:
            st.error(str(exc))
            return
        selected_period = st.selectbox(
            "Fiscal period",
            ["Latest available", *periods],
        )
        period = None if selected_period == "Latest available" else selected_period
        mode = st.radio(
            "Analysis mode",
            ["Ask Questions", "Financial Ratios", "Red Flags", "Company Comparison"],
        )

    if mode == "Ask Questions":
        _chat(client, company_id, period)
    elif mode == "Financial Ratios":
        _ratios(client, company_id, period)
    elif mode == "Red Flags":
        _red_flags(client, company_id, period)
    else:
        _comparison(client, company_ids, company_names, company_id)


def _chat(
    client: FinancialAnalyzerClient,
    company_id: str,
    period: str | None,
) -> None:
    st.subheader("Conversational Q&A")
    history = st.session_state.setdefault(
        f"financial_chat_history:{company_id}", []
    )
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                _citations(message["citations"])

    question = st.chat_input("Ask a question about the selected company")
    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                response = client.ask_question(company_id, question, period)
                answer = response.get("answer", "")
                citations = response.get("citations", [])
                st.markdown(answer)
                _citations(citations)
                history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                )
            except AnalyzerAPIError as exc:
                st.error(str(exc))


def _ratios(
    client: FinancialAnalyzerClient,
    company_id: str,
    period: str | None,
) -> None:
    st.subheader("Financial ratios")
    try:
        result = client.get_ratios(company_id, period)
    except AnalyzerAPIError as exc:
        st.error(str(exc))
        return
    sections = {
        "Liquidity": ("current_ratio", "quick_ratio", "operating_cash_flow_ratio"),
        "Leverage": ("debt_to_equity", "debt_to_assets", "interest_coverage"),
        "Profitability": (
            "gross_margin", "operating_margin", "net_margin", "roa", "roe",
        ),
        "Efficiency": ("dso", "inventory_days", "asset_turnover"),
    }
    returned = {
        metric["metric_name"]: metric for metric in result.get("metrics", [])
    }
    for section, names in sections.items():
        st.markdown(f"### {section}")
        columns = st.columns(3)
        for index, name in enumerate(names):
            metric = returned.get(name)
            if metric is None or metric.get("value") is None:
                value = "Not available"
            else:
                value = f"{metric['value']} {metric['display_unit']}"
            columns[index % len(columns)].metric(name.replace("_", " ").title(), value)
            if metric and metric.get("citations"):
                columns[index % len(columns)].caption(
                    "Sources: " + " ".join(metric["citations"])
                )
            if metric and metric.get("quality_status") != "VALID":
                columns[index % len(columns)].caption(
                    ", ".join(metric.get("warnings", [])) or "Insufficient evidence"
                )


def _red_flags(
    client: FinancialAnalyzerClient,
    company_id: str,
    period: str | None,
) -> None:
    st.subheader("Red flags")
    try:
        result = client.get_red_flags(company_id, period)
    except AnalyzerAPIError as exc:
        st.error(str(exc))
        return
    for flag in result.get("red_flags", []):
        st.write(
            f"**{flag['flag_type'].replace('_', ' ').title()}** — "
            f"{flag['status'].replace('_', ' ').title()}"
        )
        if flag.get("explanation_code"):
            st.caption(flag["explanation_code"])
        if flag.get("warnings"):
            st.warning(", ".join(flag["warnings"]))
        _citations(flag.get("citations", []))
    if not result.get("red_flags"):
        st.info("No red-flag results are available for this period.")


def _comparison(
    client: FinancialAnalyzerClient,
    company_ids: list[str],
    company_names: dict[str, str],
    selected_company: str,
) -> None:
    st.subheader("Company comparison")
    if len(company_ids) < 2:
        st.info("Upload statements for a second company to compare companies.")
        return
    first = st.selectbox(
        "Company A",
        company_ids,
        index=company_ids.index(selected_company),
        format_func=lambda value: company_names[value],
    )
    second_choices = [value for value in company_ids if value != first]
    second = st.selectbox(
        "Company B",
        second_choices,
        format_func=lambda value: company_names[value],
    )
    try:
        periods_a = client.get_periods(first)
        periods_b = client.get_periods(second)
    except AnalyzerAPIError as exc:
        st.error(str(exc))
        return
    period_a_label = st.selectbox(
        "Period A", ["Latest available", *periods_a]
    )
    period_b_label = st.selectbox(
        "Period B", ["Latest available", *periods_b]
    )
    metrics = st.multiselect(
        "Metrics (leave blank to compare supported ratios)",
        [
            "current_ratio", "quick_ratio", "operating_cash_flow_ratio",
            "debt_to_equity", "debt_to_assets", "interest_coverage",
            "gross_margin", "operating_margin", "net_margin", "roa", "roe",
            "dso", "inventory_days", "asset_turnover",
        ],
    )
    if not st.button("Compare"):
        return
    try:
        result = client.compare_companies(
            first,
            second,
            metrics or None,
            None if period_a_label == "Latest available" else period_a_label,
            None if period_b_label == "Latest available" else period_b_label,
        )
    except AnalyzerAPIError as exc:
        st.error(str(exc))
        return
    for item in result.get("metrics", []):
        left = item.get("company_a")
        right = item.get("company_b")
        st.write(
            f"**{item['metric_name'].replace('_', ' ').title()}** — "
            f"{_comparison_value(left)} | {_comparison_value(right)} | "
            f"{item['comparison_status']}"
        )
    for warning in result.get("warnings", []):
        st.warning(f"{warning['code']}: {warning['message']}")


def _comparison_value(value: dict[str, Any] | None) -> str:
    if value is None:
        return "Not available"
    unit = value.get("display_unit") or value.get("unit") or ""
    text = f"{value.get('value')} {unit}".strip()
    citations = value.get("citations", [])
    return text + (f" ({' '.join(citations)})" if citations else "")


def _citations(citations: list[str]) -> None:
    if citations:
        st.caption("Sources: " + " ".join(citations))


if __name__ == "__main__":
    main()

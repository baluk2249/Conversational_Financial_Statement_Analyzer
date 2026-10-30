from __future__ import annotations

import httpx
import pytest

from frontend.api_client import AnalyzerAPIError, FinancialAnalyzerClient


def test_client_health_question_and_upload() -> None:
    requests: list[httpx.Request] = []
    upload_timeouts: list[dict[str, float] | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/documents/upload":
            upload_timeouts.append(request.extensions.get("timeout"))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/questions/ask":
            return httpx.Response(200, json={"answer": "grounded"})
        return httpx.Response(200, json={"status": "PROCESSED"})

    client = FinancialAnalyzerClient(
        "http://test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert client.health()["status"] == "healthy"
    assert client.ask_question("abc", "What was revenue?")["answer"] == "grounded"
    assert client.upload_document("report.csv", b"data")["status"] == "PROCESSED"
    assert [request.url.path for request in requests] == [
        "/health",
        "/questions/ask",
        "/documents/upload",
    ]
    assert b"report.csv" in requests[-1].read()
    assert upload_timeouts == [
        {"connect": 180, "read": 180, "write": 180, "pool": 180}
    ]


def test_client_translates_api_and_connection_errors() -> None:
    server_error_client = FinancialAnalyzerClient(
        "http://test",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(500, json={"detail": "internal path"})
            )
        ),
    )
    with pytest.raises(AnalyzerAPIError, match="unexpected server error"):
        server_error_client.health()

    disconnected_client = FinancialAnalyzerClient(
        "http://test",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: (_ for _ in ()).throw(httpx.ConnectError("offline"))
            )
        ),
    )
    with pytest.raises(AnalyzerAPIError, match="API is unavailable"):
        disconnected_client.health()


def test_client_reports_request_timeout_separately() -> None:
    timed_out_client = FinancialAnalyzerClient(
        "http://test",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: (_ for _ in ()).throw(
                    httpx.ReadTimeout("request timed out")
                )
            )
        ),
    )
    with pytest.raises(AnalyzerAPIError, match="request timed out"):
        timed_out_client.health()


def test_summary_client_methods_send_scoped_requests() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"summary": "grounded"})

    client = FinancialAnalyzerClient(
        "http://test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.summarize_ratio_section("acme", "Liquidity", "FY2025")[
        "summary"
    ] == "grounded"
    assert client.summarize_red_flags("acme", "FY2025")["summary"] == "grounded"
    assert client.summarize_comparison(
        "acme", "zenith", ["current_ratio"], "FY2025", "FY2025"
    )["summary"] == "grounded"

    assert [request.url.path for request in requests] == [
        "/analysis/acme/summaries/ratios",
        "/analysis/acme/summaries/red-flags",
        "/comparison/summary",
    ]
    assert requests[0].read() == b'{"section":"Liquidity","period":"FY2025"}'
    assert requests[1].read() == b'{"period":"FY2025"}'
    assert b'"company_a":"acme"' in requests[2].read()

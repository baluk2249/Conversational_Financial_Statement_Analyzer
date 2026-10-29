from __future__ import annotations

import httpx
import pytest

from frontend.api_client import AnalyzerAPIError, FinancialAnalyzerClient


def test_client_health_question_and_upload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
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

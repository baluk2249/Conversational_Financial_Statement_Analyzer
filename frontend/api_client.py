"""HTTP client kept independent from Streamlit rendering."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx


class AnalyzerAPIError(RuntimeError):
    """User-safe API failure with an optional HTTP status."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FinancialAnalyzerClient:
    """Small synchronous client for the FastAPI application."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 30,
        upload_timeout: float = 180,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("FASTAPI_BASE_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.timeout = timeout
        self.upload_timeout = upload_timeout
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def list_companies(self) -> list[dict[str, Any]]:
        return self._request("GET", "/companies")["companies"]

    def get_periods(self, company_id: str) -> list[str]:
        return self._request(
            "GET", f"/companies/{_path_segment(company_id)}/periods"
        )["periods"]

    def upload_document(
        self,
        filename: str,
        content: bytes,
        *,
        company_name: str | None = None,
        company_id: str | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        fields = {
            key: value
            for key, value in {
                "company_name": company_name,
                "company_id": company_id,
                "period": period,
            }.items()
            if value
        }
        return self._request(
            "POST",
            "/documents/upload",
            files={"file": (filename, content)},
            data=fields,
            timeout=self.upload_timeout,
        )

    def ask_question(
        self,
        company_id: str,
        question: str,
        period: str | None = None,
        comparison_company_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/questions/ask",
            json={
                "company_id": company_id,
                "question": question,
                "period": period,
                "comparison_company_id": comparison_company_id,
            },
        )

    def get_ratios(
        self, company_id: str, period: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/analysis/{_path_segment(company_id)}/ratios",
            params={"period": period} if period else None,
        )

    def get_red_flags(
        self, company_id: str, period: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/analysis/{_path_segment(company_id)}/red-flags",
            params={"period": period} if period else None,
        )

    def summarize_ratio_section(
        self,
        company_id: str,
        section: str,
        period: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/analysis/{_path_segment(company_id)}/summaries/ratios",
            json={"section": section, "period": period},
        )

    def summarize_red_flags(
        self, company_id: str, period: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/analysis/{_path_segment(company_id)}/summaries/red-flags",
            json={"period": period},
        )

    def compare_companies(
        self,
        company_a: str,
        company_b: str,
        metrics: list[str] | None = None,
        period_a: str | None = None,
        period_b: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/comparison",
            json={
                "company_a": company_a,
                "company_b": company_b,
                "metrics": metrics,
                "period_a": period_a,
                "period_b": period_b,
            },
        )

    def summarize_comparison(
        self,
        company_a: str,
        company_b: str,
        metrics: list[str] | None = None,
        period_a: str | None = None,
        period_b: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/comparison/summary",
            json={
                "company_a": company_a,
                "company_b": company_b,
                "metrics": metrics,
                "period_a": period_a,
                "period_b": period_b,
            },
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(
                method, f"{self.base_url}{path}", **kwargs
            )
        except httpx.TimeoutException as exc:
            raise AnalyzerAPIError(
                "The analyzer API request timed out. The document may still be processing; check its status before retrying."
            ) from exc
        except httpx.ConnectError as exc:
            raise AnalyzerAPIError(
                "The analyzer API is unavailable. Check that FastAPI is running."
            ) from exc
        except httpx.HTTPError as exc:
            raise AnalyzerAPIError(
                "The request could not be completed."
            ) from exc

        if response.is_error:
            raise AnalyzerAPIError(
                _friendly_http_error(response.status_code, response),
                response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise AnalyzerAPIError(
                "The analyzer API returned an invalid response."
            ) from exc
        if not isinstance(payload, dict):
            raise AnalyzerAPIError("The analyzer API returned an invalid response.")
        return payload


def _friendly_http_error(status_code: int, response: httpx.Response) -> str:
    if status_code in (502, 503):
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, str) and detail:
            return detail
        if status_code == 503:
            return "A required analyzer service is unavailable."
    if status_code >= 500:
        return "The analyzer encountered an unexpected server error."
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    messages = {
        400: "The request is invalid.",
        404: "The requested company or period was not found.",
        413: "The uploaded document exceeds the allowed size.",
        415: "This document type is not supported.",
        422: "The document or request could not be processed.",
    }
    return messages.get(status_code, "The request could not be processed.")


def _path_segment(value: str) -> str:
    return quote(value, safe="")

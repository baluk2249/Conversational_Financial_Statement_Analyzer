"""HTTP API for the financial statement analyzer."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import Field

from application.service import (
    AnalyzerService,
    CompanyNotFoundError,
    PeriodNotFoundError,
)
from ingestion import (
    DocumentProcessingError,
    UnsupportedDocumentType,
)
from models.schemas import FinancialModel
from orchestration.errors import OrchestrationInputError
from retrieval.structured_store import StructuredStoreError

logger = logging.getLogger(__name__)


class AskQuestionRequest(FinancialModel):
    company_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    period: str | None = None
    comparison_company_id: str | None = None


class ComparisonAPIRequest(FinancialModel):
    company_a: str = Field(min_length=1)
    company_b: str = Field(min_length=1)
    metrics: list[str] | None = None
    period_a: str | None = None
    period_b: str | None = None


def create_app(service: AnalyzerService | None = None) -> FastAPI:
    analyzer = service or AnalyzerService.from_environment()
    app = FastAPI(
        title="Conversational Financial Statement Analyzer",
        version="0.1.0",
        description="Grounded financial evidence, deterministic analysis, and citations.",
    )
    app.state.analyzer = analyzer

    @app.exception_handler(CompanyNotFoundError)
    async def unknown_company_handler(_request: Any, _exc: CompanyNotFoundError):
        del _request, _exc
        return _json_error(404, "The requested company was not found.")

    @app.exception_handler(PeriodNotFoundError)
    async def unknown_period_handler(_request: Any, _exc: PeriodNotFoundError):
        del _request, _exc
        return _json_error(404, "The requested fiscal period was not found.")

    @app.exception_handler(UnsupportedDocumentType)
    async def unsupported_document_handler(
        _request: Any, exc: UnsupportedDocumentType
    ):
        del _request
        return _json_error(415, str(exc))

    @app.exception_handler(DocumentProcessingError)
    async def document_error_handler(
        _request: Any, exc: DocumentProcessingError
    ):
        del _request
        return _json_error(422, str(exc))

    @app.exception_handler(OrchestrationInputError)
    async def orchestration_input_handler(
        _request: Any, exc: OrchestrationInputError
    ):
        del _request
        return _json_error(400, str(exc))

    @app.exception_handler(StructuredStoreError)
    async def structured_store_error_handler(
        _request: Any, exc: StructuredStoreError
    ):
        del _request
        logger.error(
            "Structured store operation failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return _json_error(503, "The structured financial store is unavailable.")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return analyzer.health()

    @app.get("/companies")
    def companies() -> dict[str, Any]:
        return {"companies": analyzer.list_companies()}

    @app.get("/companies/{company_id}/periods")
    def company_periods(company_id: str) -> dict[str, Any]:
        return {"company_id": company_id, "periods": analyzer.get_periods(company_id)}

    @app.post("/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        company_name: str | None = Form(default=None),
        company_id: str | None = Form(default=None),
        period: str | None = Form(default=None),
    ) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=422, detail="A file name is required.")
        content = await file.read(20 * 1024 * 1024 + 1)
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="The uploaded document exceeds the 20 MB limit.",
            )
        return await run_in_threadpool(
            analyzer.ingest_document,
            file.filename,
            content,
            company_name=company_name,
            company_id=company_id,
            period=period,
        )

    @app.post("/questions/ask")
    def ask_question(request: AskQuestionRequest) -> dict[str, Any]:
        return analyzer.ask(
            request.company_id,
            request.question,
            request.period,
            request.comparison_company_id,
        )

    @app.get("/analysis/{company_id}/ratios")
    def ratios(
        company_id: str,
        period: str | None = None,
    ) -> dict[str, Any]:
        return analyzer.get_ratios(company_id, period)

    @app.get("/analysis/{company_id}/red-flags")
    def red_flags(
        company_id: str,
        period: str | None = None,
    ) -> dict[str, Any]:
        return analyzer.get_red_flags(company_id, period)

    @app.post("/comparison")
    def compare(request: ComparisonAPIRequest) -> dict[str, Any]:
        return analyzer.compare(
            request.company_a,
            request.company_b,
            metrics=request.metrics,
            period_a=request.period_a,
            period_b=request.period_b,
        )

    return app


def _json_error(status_code: int, detail: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status_code, content={"detail": detail})


app = create_app()

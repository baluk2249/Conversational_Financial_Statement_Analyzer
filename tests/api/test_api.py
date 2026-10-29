from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from openpyxl import Workbook
from pypdf import PdfWriter

from agents.narrative_qa.agent import NarrativeQAAgent
from api.main import create_app
from application.service import AnalyzerService, LazyFaissNarrativeStore
from retrieval.structured_store import SQLiteFinancialStore


CSV = """company_id,company_name,fiscal_year,statement_type,line_item,value,currency,unit,period_start,period_end,fiscal_year_end
acme,Acme Ltd,FY2025,income_statement,revenue,1000,GBP,absolute,2025-01-01,2025-12-31,2025-12-31
acme,Acme Ltd,FY2025,balance_sheet,current_assets,150,GBP,absolute,2025-01-01,2025-12-31,2025-12-31
acme,Acme Ltd,FY2025,balance_sheet,current_liabilities,100,GBP,absolute,2025-01-01,2025-12-31,2025-12-31
zenith,Zenith Ltd,FY2025,income_statement,revenue,1200,GBP,absolute,2025-01-01,2025-12-31,2025-12-31
zenith,Zenith Ltd,FY2025,balance_sheet,current_assets,200,GBP,absolute,2025-01-01,2025-12-31,2025-12-31
zenith,Zenith Ltd,FY2025,balance_sheet,current_liabilities,100,GBP,absolute,2025-01-01,2025-12-31,2025-12-31
"""


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_service(tmp_path)))


def _service(tmp_path: Path) -> AnalyzerService:
    return AnalyzerService(
        SQLiteFinancialStore(":memory:"),
        LazyFaissNarrativeStore(tmp_path / "faiss"),
        narrative_agent=NarrativeQAAgent(),
    )


def _upload(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("sample.csv", CSV.encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["facts_stored"] == 6


def test_health_companies_and_periods(tmp_path) -> None:
    client = _client(tmp_path)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["components"]["structured_store"] == "healthy"
    assert health.json()["components"]["faiss"] == "not_initialized"

    _upload(client)
    assert client.get("/companies").json()["companies"] == [
        {"company_id": "acme", "company_name": "Acme Ltd"},
        {"company_id": "zenith", "company_name": "Zenith Ltd"},
    ]
    assert client.get("/companies/acme/periods").json()["periods"] == ["FY2025"]


def test_upload_rejects_unsupported_and_malformed_documents(tmp_path) -> None:
    client = _client(tmp_path)
    unsupported = client.post(
        "/documents/upload",
        files={"file": ("sample.doc", b"not supported", "application/octet-stream")},
    )
    assert unsupported.status_code == 415

    malformed = client.post(
        "/documents/upload",
        files={"file": ("sample.csv", b"foo,bar\n1,2\n", "text/csv")},
    )
    assert malformed.status_code == 422


def test_xlsx_upload_and_pdf_narrative_scope(tmp_path) -> None:
    client = _client(tmp_path)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "P&L"
    worksheet.append(
        ["company_name", "fiscal_year", "line_item", "value", "currency"]
    )
    worksheet.append(["Acme Ltd", "FY2025", "revenue", 1000, "GBP"])
    data = BytesIO()
    workbook.save(data)

    xlsx = client.post(
        "/documents/upload",
        files={
            "file": (
                "statements.xlsx",
                data.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert xlsx.status_code == 200
    assert xlsx.json()["facts_stored"] == 1
    assert xlsx.json()["company_ids"] == ["acme-ltd"]
    assert xlsx.json()["periods"] == ["FY2025"]

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    pdf_data = BytesIO()
    writer.write(pdf_data)
    missing_scope = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", pdf_data.getvalue(), "application/pdf")},
    )
    assert missing_scope.status_code == 422

    pdf = client.post(
        "/documents/upload",
        data={"company_name": "Acme Ltd", "period": "FY2025"},
        files={"file": ("report.pdf", pdf_data.getvalue(), "application/pdf")},
    )
    assert pdf.status_code == 200
    assert "NO_PDF_TABLES_EXTRACTED" in pdf.json()["warnings"]
    assert "NO_NORMALIZED_FINANCIAL_FACTS_FOUND" in pdf.json()["warnings"]


def test_narrative_only_company_is_listed_without_fabricated_facts(tmp_path) -> None:
    class InMemoryNarrativeStore:
        def __init__(self) -> None:
            self.chunks = []

        def add_chunks(self, chunks) -> None:
            self.chunks.extend(chunks)

        def search(self, *args, **kwargs):
            return []

    narrative_store = InMemoryNarrativeStore()
    service = AnalyzerService(
        SQLiteFinancialStore(":memory:"),
        narrative_store,
        narrative_agent=NarrativeQAAgent(),
    )
    result = service.ingest_document(
        "notes.csv",
        b"company_id,company_name,fiscal_year,narrative\nacme,Acme Ltd,FY2025,Management discussed customer demand.\n",
    )
    assert result["facts_stored"] == 0
    assert result["narrative_chunks"] == 1
    assert service.list_companies() == [
        {"company_id": "acme", "company_name": "Acme Ltd"}
    ]
    assert service.get_periods("acme") == ["FY2025"]
    assert "NO_STRUCTURED_FINANCIAL_FACTS_FOUND" in result["warnings"]


def test_question_ratio_red_flag_and_comparison_endpoints(tmp_path) -> None:
    client = _client(tmp_path)
    _upload(client)

    question = client.post(
        "/questions/ask",
        json={
            "company_id": "acme",
            "question": "What was revenue in FY2025?",
            "period": "FY2025",
        },
    )
    assert question.status_code == 200
    assert question.json()["status"] == "OK"
    citation = "[Income Statement | FY2025 | revenue | Sheet: CSV | Cell: E2:F2]"
    assert citation in question.json()["answer"]
    assert question.json()["citations"] == [
        citation
    ]

    ratios = client.get("/analysis/acme/ratios", params={"period": "FY2025"})
    assert ratios.status_code == 200
    current_ratio = next(
        item for item in ratios.json()["metrics"]
        if item["metric_name"] == "current_ratio"
    )
    assert current_ratio["value"] == "1.5000"
    assert current_ratio["formula_version"] == "current_ratio_v1"
    assert len(current_ratio["citations"]) == 2

    flags = client.get("/analysis/acme/red-flags", params={"period": "FY2025"})
    assert flags.status_code == 200
    assert {item["flag_type"] for item in flags.json()["red_flags"]} == {
        "accrual_quality_gap",
        "dso_climbing",
        "margin_compression",
        "leverage_stress",
        "inventory_build_up",
        "liquidity_deterioration",
    }

    comparison = client.post(
        "/comparison",
        json={
            "company_a": "acme",
            "company_b": "zenith",
            "metrics": ["current_ratio"],
            "period_a": "FY2025",
            "period_b": "FY2025",
        },
    )
    assert comparison.status_code == 200
    assert comparison.json()["metrics"][0]["comparison_status"] == "COMPARABLE"
    assert comparison.json()["metrics"][0]["company_a"]["citations"]
    assert comparison.json()["metrics"][0]["company_b"]["citations"]


def test_insufficient_evidence_and_unknown_scope_are_explicit(tmp_path) -> None:
    client = _client(tmp_path)
    _upload(client)

    missing = client.post(
        "/questions/ask",
        json={"company_id": "acme", "question": "What was EBITDA in FY2025?"},
    )
    assert missing.status_code == 200
    assert missing.json()["status"] == "INSUFFICIENT_EVIDENCE"

    unknown_company = client.get("/companies/missing/periods")
    assert unknown_company.status_code == 404
    unknown_period = client.get(
        "/analysis/acme/ratios", params={"period": "FY2030"}
    )
    assert unknown_period.status_code == 404

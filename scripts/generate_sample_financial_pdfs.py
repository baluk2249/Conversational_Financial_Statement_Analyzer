"""Generate selectable-text synthetic financial-statement PDF fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "tests" / "fixtures" / "pdfs"
COMPANY = "ABC DEMO LTD"
PERIOD_HEADERS = ["Line Item", "FY2025", "FY2024"]
COMPANY_PERIOD_HEADERS = ["Line Item", "FY2025", "FY2024", "FY2023"]

INCOME_ROWS = [
    ["Revenue", "120.0", "100.0"],
    ["Cost of Goods Sold", "(72.0)", "(58.0)"],
    ["Gross Profit", "48.0", "42.0"],
    ["Operating Profit", "18.0", "20.0"],
    ["Interest Expense", "(3.0)", "(2.0)"],
    ["Net Income", "11.0", "14.0"],
]

BALANCE_ROWS = [
    ["Current Assets", "52.0", "50.0"],
    ["Inventory", "16.0", "10.0"],
    ["Accounts Receivable", "18.0", "12.0"],
    ["Current Liabilities", "40.0", "30.0"],
    ["Total Assets", "150.0", "130.0"],
    ["Total Debt", "60.0", "40.0"],
    ["Total Equity", "52.0", "50.0"],
]

CASH_FLOW_ROWS = [
    ["Net Income", "11.0", "14.0"],
    ["Operating Cash Flow", "8.0", "18.0"],
    ["Investing Cash Flow", "(12.0)", "(10.0)"],
    ["Financing Cash Flow", "5.0", "1.0"],
]

COMMENTARY = (
    "This page contains synthetic test commentary only. During FY2025, "
    "revenue increased. Operating expenses increased. Inventory levels "
    "increased, and receivables increased. Operating cash flow declined."
)

COMPANY_SAMPLES: dict[str, dict[str, Any]] = {
    "sample_company_northstar_financials.pdf": {
        "name": "NORTHSTAR DEMO LTD",
        "income": [
            ["Revenue", "145.0", "126.0", "110.0"],
            ["Cost of Goods Sold", "(87.0)", "(75.6)", "(66.0)"],
            ["Gross Profit", "58.0", "50.4", "44.0"],
            ["Operating Profit", "21.0", "18.5", "15.0"],
            ["Interest Expense", "(3.5)", "(3.0)", "(2.7)"],
            ["Net Income", "13.5", "11.5", "9.5"],
        ],
        "balance": [
            ["Current Assets", "61.0", "55.0", "50.0"],
            ["Inventory", "15.0", "14.0", "13.0"],
            ["Accounts Receivable", "21.0", "17.0", "15.0"],
            ["Current Liabilities", "43.0", "38.0", "35.0"],
            ["Total Assets", "172.0", "154.0", "140.0"],
            ["Total Debt", "66.0", "58.0", "52.0"],
            ["Total Equity", "62.0", "56.0", "51.0"],
        ],
        "cash_flow": [
            ["Net Income", "13.5", "11.5", "9.5"],
            ["Operating Cash Flow", "17.0", "14.0", "12.0"],
            ["Investing Cash Flow", "(14.0)", "(12.0)", "(10.0)"],
            ["Financing Cash Flow", "(2.0)", "(1.0)", "(1.0)"],
        ],
        "income_period": "For the year ended 31 December 2025, 2024 and 2023",
        "balance_period": "As at 31 December 2025, 2024 and 2023",
        "cash_flow_period": "For the year ended 31 December 2025, 2024 and 2023",
        "commentary": (
            "Synthetic Northstar demo commentary: revenue and operating cash "
            "flow increased in FY2025. Inventory and receivables also increased."
        ),
    },
    "sample_company_cedar_financials.pdf": {
        "name": "CEDAR DEMO PLC",
        "income": [
            ["Revenue", "98.0", "105.0", "120.0"],
            ["Cost of Goods Sold", "(61.0)", "(63.0)", "(70.0)"],
            ["Gross Profit", "37.0", "42.0", "50.0"],
            ["Operating Profit", "12.0", "16.0", "22.0"],
            ["Interest Expense", "(2.0)", "(1.8)", "(1.5)"],
            ["Net Income", "7.0", "10.5", "15.0"],
        ],
        "balance": [
            ["Current Assets", "40.0", "47.0", "50.0"],
            ["Inventory", "18.0", "12.0", "11.0"],
            ["Accounts Receivable", "19.0", "16.0", "15.0"],
            ["Current Liabilities", "40.0", "34.0", "30.0"],
            ["Total Assets", "128.0", "133.0", "130.0"],
            ["Total Debt", "48.0", "43.0", "40.0"],
            ["Total Equity", "45.0", "51.0", "60.0"],
        ],
        "cash_flow": [
            ["Net Income", "7.0", "10.5", "15.0"],
            ["Operating Cash Flow", "3.0", "13.0", "20.0"],
            ["Investing Cash Flow", "(8.0)", "(11.0)", "(14.0)"],
            ["Financing Cash Flow", "(4.0)", "(2.0)", "(1.0)"],
        ],
        "income_period": "For the year ended 31 December 2025, 2024 and 2023",
        "balance_period": "As at 31 December 2025, 2024 and 2023",
        "cash_flow_period": "For the year ended 31 December 2025, 2024 and 2023",
        "commentary": (
            "Synthetic Cedar demo commentary: revenue and operating profit "
            "declined in FY2025. Receivables and inventory increased, while "
            "operating cash flow declined."
        ),
    },
    "sample_company_orbit_financials.pdf": {
        "name": "ORBIT DEMO INC",
        "income": [
            ["Revenue", "176.0", "150.0", "130.0"],
            ["Cost of Goods Sold", "(98.0)", "(87.0)", "(78.0)"],
            ["Gross Profit", "78.0", "63.0", "52.0"],
            ["Operating Profit", "29.0", "24.0", "20.0"],
            ["Interest Expense", "(4.0)", "(3.5)", "(3.0)"],
            ["Net Income", "18.0", "15.0", "12.0"],
        ],
        "balance": [
            ["Current Assets", "73.0", "64.0", "58.0"],
            ["Inventory", "22.0", "18.0", "16.0"],
            ["Accounts Receivable", "25.0", "21.0", "18.0"],
            ["Current Liabilities", "49.0", "44.0", "40.0"],
            ["Total Assets", "205.0", "184.0", "165.0"],
            ["Total Debt", "71.0", "65.0", "58.0"],
            ["Total Equity", "79.0", "70.0", "62.0"],
        ],
        "cash_flow": [
            ["Net Income", "18.0", "15.0", "12.0"],
            ["Operating Cash Flow", "22.0", "19.0", "17.0"],
            ["Investing Cash Flow", "(18.0)", "(15.0)", "(13.0)"],
            ["Financing Cash Flow", "(1.0)", "(2.0)", "(2.0)"],
        ],
        "income_period": "For the year ended 31 December 2025, 2024 and 2023",
        "balance_period": "As at 31 December 2025, 2024 and 2023",
        "cash_flow_period": "For the year ended 31 December 2025, 2024 and 2023",
        "commentary": (
            "Synthetic Orbit demo commentary: revenue, operating profit, and "
            "operating cash flow increased in FY2025."
        ),
    },
}


def _styles() -> tuple[ParagraphStyle, ParagraphStyle, ParagraphStyle]:
    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "FixtureTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )
    notice = ParagraphStyle(
        "SyntheticNotice",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=3 * mm,
    )
    body = ParagraphStyle(
        "FixtureBody",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
    )
    return title, notice, body


def _statement_page(
    title_text: str,
    rows: Sequence[Sequence[str]],
    company: str = COMPANY,
    *,
    period_headers: Sequence[str] = PERIOD_HEADERS,
    period_context: str | None = None,
) -> list[object]:
    title_style, notice_style, _ = _styles()
    table_data = [period_headers, *rows]
    value_column_width = (170 * mm - 105 * mm) / (len(period_headers) - 1)
    table = Table(
        table_data,
        colWidths=[105 * mm, *([value_column_width] * (len(period_headers) - 1))],
        repeatRows=1,
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172B4D")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#78879A")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F6F8FA")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return [
        Paragraph("SYNTHETIC TEST DATA ONLY", notice_style),
        Paragraph(company, notice_style),
        Paragraph(title_text, title_style),
        *(
            [Paragraph(period_context, notice_style)]
            if period_context
            else []
        ),
        Paragraph("GBP millions", notice_style),
        Spacer(1, 4 * mm),
        table,
    ]


def _build_pdf(filename: str, pages: Sequence[Sequence[object]]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT_DIRECTORY / filename),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=filename.removesuffix(".pdf"),
        author="Synthetic fixture generator",
        subject="Synthetic test data only",
    )
    story: list[object] = []
    for page_index, page in enumerate(pages):
        if page_index:
            story.append(PageBreak())
        story.extend(page)
    document.build(story)


def _generate_complete() -> None:
    title_style, notice_style, body_style = _styles()
    pages = [
        _statement_page("CONSOLIDATED INCOME STATEMENT", INCOME_ROWS),
        _statement_page("CONSOLIDATED BALANCE SHEET", BALANCE_ROWS),
        _statement_page("CONSOLIDATED CASH FLOW STATEMENT", CASH_FLOW_ROWS),
        [
            Paragraph("SYNTHETIC TEST DATA ONLY", notice_style),
            Paragraph("MANAGEMENT COMMENTARY", title_style),
            Paragraph(COMMENTARY, body_style),
        ],
    ]
    _build_pdf("sample_complete_financials.pdf", pages)


def _generate_ambiguous() -> None:
    _, notice_style, title_style = _styles()
    data = [
        ["Category", "Value", "Change"],
        ["Segment A", "120.0", "8.0"],
        ["Segment B", "72.0", "(3.0)"],
        ["Segment C", "48.0", "11.0"],
    ]
    table = Table(data, colWidths=[70 * mm, 45 * mm, 45 * mm], hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#78879A")),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    _build_pdf(
        "sample_ambiguous_table.pdf",
        [[
            Paragraph("SYNTHETIC TEST DATA ONLY", notice_style),
            Paragraph("Unclassified Numeric Table", title_style),
            Spacer(1, 4 * mm),
            table,
        ]],
    )


def _generate_company_sample(filename: str, sample: dict[str, Any]) -> None:
    title_style, notice_style, body_style = _styles()
    company = str(sample["name"])
    pages = [
        _statement_page(
            "CONSOLIDATED INCOME STATEMENT",
            sample["income"],
            company,
            period_headers=COMPANY_PERIOD_HEADERS,
            period_context=str(sample["income_period"]),
        ),
        _statement_page(
            "CONSOLIDATED BALANCE SHEET",
            sample["balance"],
            company,
            period_headers=COMPANY_PERIOD_HEADERS,
            period_context=str(sample["balance_period"]),
        ),
        _statement_page(
            "CONSOLIDATED CASH FLOW STATEMENT",
            sample["cash_flow"],
            company,
            period_headers=COMPANY_PERIOD_HEADERS,
            period_context=str(sample["cash_flow_period"]),
        ),
        [
            Paragraph("SYNTHETIC TEST DATA ONLY", notice_style),
            Paragraph(company, notice_style),
            Paragraph("MANAGEMENT COMMENTARY", title_style),
            Paragraph(str(sample["commentary"]), body_style),
        ],
    ]
    _build_pdf(filename, pages)


def generate_pdfs() -> None:
    _generate_complete()
    _build_pdf(
        "sample_income_statement.pdf",
        [_statement_page("CONSOLIDATED INCOME STATEMENT", INCOME_ROWS)],
    )
    _build_pdf(
        "sample_balance_sheet.pdf",
        [_statement_page("CONSOLIDATED BALANCE SHEET", BALANCE_ROWS)],
    )
    _build_pdf(
        "sample_cash_flow.pdf",
        [_statement_page("CONSOLIDATED CASH FLOW STATEMENT", CASH_FLOW_ROWS)],
    )

    missing_rows = [
        ["Revenue", "120.0", ""],
        ["Cost of Goods Sold", "", "(58.0)"],
        ["Gross Profit", "48.0", ""],
        ["Operating Profit", "", "20.0"],
        ["Interest Expense", "(3.0)", ""],
        ["Net Income", "", "14.0"],
    ]
    _build_pdf(
        "sample_missing_values.pdf",
        [_statement_page("CONSOLIDATED INCOME STATEMENT", missing_rows)],
    )
    _build_pdf(
        "sample_units_millions.pdf",
        [_statement_page("CONSOLIDATED INCOME STATEMENT", INCOME_ROWS)],
    )
    _generate_ambiguous()
    for filename, sample in COMPANY_SAMPLES.items():
        _generate_company_sample(filename, sample)


if __name__ == "__main__":
    generate_pdfs()

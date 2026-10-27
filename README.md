# Conversational Financial Statement Analyzer

A GenAI tool for turning company financial statements into grounded, queryable
insights—without manual spreadsheet work. Users upload profit and loss (P&L),
balance sheet, and cash flow statements, then ask questions in plain English.

The analyzer computes financial metrics from normalized statement data,
identifies changes and potential red flags, and explains its findings with
citations to the source figures. It can also compare two companies while
calling out important differences in reporting period or currency.

## Objective

Make financial statements easier to explore and understand by combining
document retrieval, deterministic calculations, and grounded conversational
answers.

## What it does

- Ingests company P&L, balance sheet, and cash flow statements across periods.
- Normalizes statement line items and computes liquidity, leverage, and
  profitability ratios.
- Detects period-over-period trends and six potential red-flag patterns.
- Answers questions using computed facts and relevant narrative disclosures.
- Cites source figures for every financial number in a response.
- Declines to provide a figure when it cannot be supported by the uploaded
  statements.
- Optionally compares two companies and surfaces fiscal-year-end or currency
  mismatches before drawing comparisons.

## AI agents

### 1. Trend & Red Flag Agent

Computes period-over-period changes across available ratios and evaluates these
six patterns:

1. **Accrual-quality gap** — earnings are not converting into operating cash
   flow as expected.
2. **DSO climbing** — days sales outstanding is increasing, which may indicate
   slower collections.
3. **Margin compression** — gross, operating, or net margins are declining.
4. **Leverage stress** — debt or interest burden is increasing relative to
   earnings or equity.
5. **Inventory build-up** — inventory is growing faster than sales or is
   otherwise inconsistent with the available periods.
6. **Liquidity deterioration** — short-term liquidity measures are weakening.

These are indicators for further review, not conclusions about misconduct or
financial distress. A flag should include the periods and figures supporting
it, and should be qualified when required inputs are missing.

### 2. Retrieval / Grounding Agent

Indexes normalized statements and narrative notes in ChromaDB. For each
question, it assembles a grounded context block containing relevant source
text and computed facts. Numeric calculations should come from structured
statement data and deterministic tools rather than language-model estimation.

### 3. Narrative & Q&A Agent

Provides the conversational interface. It answers from the supplied grounded
context only, attaches a citation to every financial figure, and declines
unsupported numeric questions instead of inventing or extrapolating a value.

### 4. Comparison Agent

Compares two companies side by side using available, aligned metrics. It checks
fiscal-year-end and currency compatibility and clearly surfaces mismatches
rather than presenting an unqualified comparison.

## Key metrics

The specific ratios depend on which statement line items are available. Typical
examples include:

| Category | Example metrics |
| --- | --- |
| Liquidity | Current ratio, quick ratio |
| Leverage | Debt-to-equity, debt-to-assets, interest coverage |
| Profitability | Gross margin, operating margin, net margin, return on assets |
| Cash flow | Operating cash flow, free cash flow, operating-cash-flow-to-net-income |
| Working capital | Days sales outstanding (DSO), inventory trend |

Each reported metric should identify its formula, reporting period, currency
where applicable, and the statement figures used to calculate it. If required
inputs are absent or not comparable, the system should say so rather than
silently substituting assumptions.

## Grounding and citations

Financial answers must be traceable to uploaded material. Citations should
identify the company, statement, reporting period, and source line item (and
page or document location when available). Derived ratios and trends should
cite their input figures and show the calculation or formula where useful.

The response workflow is:

1. Retrieve relevant statement data and narrative disclosures.
2. Compute requested values and trends using reliable numeric tools.
3. Construct an answer from the retrieved evidence and computed facts.
4. Attach source citations to every financial figure.
5. Decline or qualify claims that the available evidence cannot support.

## Typical user questions

- “How did the current ratio change over the last three periods?”
- “Why did operating cash flow diverge from net income?”
- “Are receivables increasing faster than revenue?”
- “What are the main margin and leverage trends?”
- “Compare the two companies’ profitability, and flag any differences in
  currency or fiscal year-end.”

## Learning takeaways

- Retrieval-augmented generation (RAG) over financial documents.
- Tool use for reliable numeric computation.
- Hallucination control for financial figures.
- Grounding and citation techniques.
- Financial-ratio calculation and interpretation.
- Cross-company comparison with period and currency checks.

## Important note

This tool is intended to support financial-statement analysis and education. Its
outputs are not investment, legal, tax, or accounting advice. Red flags are
signals to investigate, not definitive judgments; users should verify source
documents and consult qualified professionals for consequential decisions.

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

Indexes normalized statements and narrative notes in FAISS. For each
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

## Retrieval / Grounding MVP

The retrieval package keeps exact financial facts in SQLite and narrative passages
in a FAISS index. It does not use vector search for numeric facts or ask a language
model to calculate ratios. Pydantic evidence contracts preserve source locators,
and the ratio layer returns a formula version plus its input fact IDs. Trend and
red-flag results are read through a provider interface so the owning agent can
implement them independently.

Requirements: Python 3.11 or newer. From the repository root, install the package
and test dependencies, then run the test suite:

```powershell
python -m pip install -e ".[dev,semantic]"
python -m pytest
```

Run the explicitly synthetic demonstration with:

```powershell
python examples/retrieval_demo.py
```

The demo writes local SQLite and FAISS data under `.data/`. The first narrative
embedding may download the configured Model2Vec model. Production ingestion
should supply normalized `FinancialFact` and `DocumentChunk` records from the
upstream PDF/XLSX/CSV parsers; this package intentionally does not duplicate them.

The main service is `RetrievalGroundingAgent.retrieve_context(question,
company_id, period=None, top_k=5)`. Exact lookups are company- and period-scoped;
unsupported figures produce `INSUFFICIENT_EVIDENCE`. Ratios not already stored
are computed deterministically with `Decimal`, and incompatible currencies or
units make the result unavailable with an explicit warning.

## Trend & Red Flag Agent

The deterministic Trend & Red Flag Agent accepts shared `FinancialFact` records
and optionally existing `ComputedMetric` records. It calculates supported
liquidity, leverage, profitability, and efficiency ratios with `Decimal`, then
compares adjacent periods and evaluates the six configured red-flag families.
It does not use an LLM for formulas, period deltas, or rule decisions.

```python
from agents.trend_red_flag import TrendRedFlagAgent

analysis = TrendRedFlagAgent().analyse(
    company_id="abc-ltd",
    facts=normalized_facts,
    periods=["FY2024", "FY2025"],  # caller-provided chronological order
)
```

`analysis` returns the company-scoped source facts, formula-versioned metrics,
trend results, and red flags. A trend's `supporting_refs` identify its computed
metrics; those metrics identify their input fact IDs. Flags point to their
supporting trend/metric IDs or, for the accrual and inventory detectors, the
exact facts used. Metric and trend quality statuses expose missing or
incompatible inputs instead of filling them with zero or assumed values.

Rule versions and thresholds are loaded from
[`config/red_flag_rules.yaml`](./config/red_flag_rules.yaml). The checked-in
defaults are direction-only (`0` minimum adverse change), not industry
benchmarks; tune those values in configuration for an explicitly selected
policy. A missing threshold leaves the affected detector at
`INSUFFICIENT_EVIDENCE`. Disabled detectors return `NOT_APPLICABLE`.

Run the local synthetic example with:

```powershell
python examples/trend_red_flag_demo.py
```

Run the focused agent tests with:

```powershell
python -m pytest tests/trend_red_flag
```

## Narrative & Q&A Agent

`NarrativeQAAgent` consumes the shared `AnswerEvidence` model. It uses only
company-scoped facts, calculated metrics, trends, red flags, narrative passages,
and warnings in that bundle. Citations are built from source metadata and
calculated metrics are cited through their input fact references. A post-
generation validator rejects unsupported numeric claims, unresolved citations,
and causal statements that are not supported by retrieved narrative evidence.

The deterministic formatter handles exact facts, metrics, trends, red flags,
and insufficient-evidence responses if no model is configured or a provider
request fails. Forecast/estimate requests and comparison questions are refused
deterministically: the current `AnswerEvidence` contract does not include
verified comparison bundles or forecast inputs.

Run the synthetic example (no API key or network request required):

```powershell
python examples/narrative_qa_demo.py
```

For optional generation, inject an `LLMClient`. An OpenAI-compatible adapter
is included and reads credentials only from environment variables:

```python
import os

from agents.narrative_qa import NarrativeQAAgent
from llm.providers import OpenAICompatibleClient

agent = NarrativeQAAgent(
    llm_client=OpenAICompatibleClient.from_environment()
)
answer = agent.answer(answer_evidence)
```

Set `OPENAI_API_KEY`; optionally set `OPENAI_BASE_URL` and `OPENAI_MODEL`.
Configuring this adapter sends the targeted evidence block to that endpoint.
Without an explicitly supplied client the agent uses local deterministic
formatting and makes no network requests. Validate the project changes with:

```powershell
python -m pytest tests/narrative_qa
```

## Comparison Agent

`ComparisonAgent` accepts two separate shared `AnswerEvidence` bundles and
compares only company-scoped facts or valid computed metrics. It validates
metric/formula definitions, fiscal periods and year ends, currencies, and unit
scales before producing a deterministic `company_b - company_a` difference.
It reports the higher value only; it does not infer that a higher value is
financially better.

Currency is never converted. A currency mismatch blocks comparisons of
monetary facts, but is surfaced as a warning while definition- and
period-compatible ratios remain comparable. Known `absolute`, `thousands`, and
`millions` scales are aligned deterministically for monetary comparisons while
the original values, units, evidence IDs, and source references remain
available in each result. Unknown or absent required metadata prevents a
comparison. Missing evidence never produces a winner. Upstream warnings are
preserved in structured result warnings. Trend questions (for example,
"which margin improved more?") compare `TrendResult` deltas only when both
periods, formula-versioned supporting metrics, and source references are
present and aligned; otherwise the result carries an explicit warning.

```python
from agents.comparison import ComparisonAgent, ComparisonRequest

result = ComparisonAgent().compare(
    company_a_evidence,
    company_b_evidence,
    ComparisonRequest(
        question="Compare liquidity.",
        metric_names=["current_ratio", "quick_ratio"],
        period="FY2025",
    ),
)
```

For a synthetic local demonstration with no API calls:

```powershell
python examples/comparison_demo.py
```

Run the comparison tests with:

```powershell
python -m pytest tests/comparison
```

## Important note

This tool is intended to support financial-statement analysis and education. Its
outputs are not investment, legal, tax, or accounting advice. Red flags are
signals to investigate, not definitive judgments; users should verify source
documents and consult qualified professionals for consequential decisions.

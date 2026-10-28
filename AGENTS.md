# AGENTS.md

## Project: Conversational Financial Statement Analyzer

This file defines the shared behavioural and integration contract for all agents in the Conversational Financial Statement Analyzer repository.

The system turns uploaded company financial statements into grounded, conversational financial insight. It must support exact figure lookup, deterministic ratio calculation, period-over-period trend analysis, red-flag detection, explanatory Q&A with citations, and optional side-by-side company comparison.

## Core Principle

> The language model explains. The calculation engine calculates. The retrieval layer proves.

All agents must preserve this separation of responsibilities.

---

## 1. System Objectives

The application should:

- ingest Profit & Loss / Income Statements, Balance Sheets, Cash Flow Statements, and relevant narrative notes;
- normalise financial line items into a consistent internal schema;
- compute financial ratios using deterministic application code;
- identify trends across reporting periods;
- evaluate six supported financial red-flag patterns;
- answer conversational questions using grounded evidence only;
- attach source provenance to every financial figure presented to the user;
- refuse or return insufficient evidence when requested figures are unavailable;
- optionally compare two companies while surfacing period, currency, unit, fiscal-year-end, or definition mismatches.

---

## 2. Agent Roster

### Trend & Red Flag Agent

**Owner:** Vinayak Kulkarni

**Mission:** Compute period-over-period changes and evaluate supported financial red flags from normalised facts and deterministic financial metrics.

Primary responsibilities:

- consume normalised financial facts;
- consume or invoke deterministic ratio calculations;
- compare ratios and financial line items across periods;
- calculate period-over-period deltas;
- identify supported red-flag patterns;
- retain references to the exact facts and calculations supporting each result;
- return `INSUFFICIENT_EVIDENCE` when a detector lacks the required periods or inputs.

Supported red-flag families:

1. accrual-quality gap;
2. DSO climbing;
3. margin compression;
4. leverage stress;
5. inventory build-up;
6. liquidity deterioration.

The agent must not invent thresholds. Detector thresholds and formula definitions must come from version-controlled application configuration.

---

### Retrieval / Grounding Agent

**Owner:** Balaraju Kotturi

**Mission:** Assemble exact, traceable evidence for downstream agents using structured financial lookup plus semantic retrieval of narrative material.

Primary responsibilities:

- identify the company, period, metrics, statement types, and narrative topics needed for a question;
- retrieve exact financial values from the structured fact store;
- retrieve existing deterministic calculations and trend/red-flag outputs;
- retrieve relevant narrative notes and commentary from a FAISS vector index;
- filter retrieval by company and fiscal period before semantic ranking whenever applicable;
- preserve source file and page/cell/paragraph locators;
- build a stable grounded-context payload;
- never introduce unsupported values;
- return `INSUFFICIENT_EVIDENCE` where required evidence is missing.

Exact financial facts must not depend on vector similarity as the source of truth.

---

### Narrative & Q&A Agent

**Owner:** Pramod Bhosale

**Mission:** Convert grounded evidence into clear conversational answers without adding unsupported financial claims.

Primary responsibilities:

- answer only from the grounded context supplied by the system;
- explain deterministic metrics in plain English;
- describe relevant trends and red flags without overstating their meaning;
- attach a citation to every financial figure;
- distinguish reported figures from calculated metrics;
- preserve currency, units, company, and fiscal-period context;
- decline numeric questions when evidence is insufficient;
- never invent a number, source, management explanation, or financial cause.

The agent may explain calculations but must not become the authoritative calculation engine.

---

### Comparison Agent

**Owner:** Sandip Shaw

**Mission:** Produce evidence-backed side-by-side company comparisons only after checking comparability.

Primary responsibilities:

- request grounded evidence independently for each company;
- align periods and metric definitions where possible;
- check fiscal-year-end, currency, units, and reporting-scope metadata;
- surface mismatches explicitly;
- compare ratios only when definitions and periods are sufficiently aligned;
- keep each company's citations isolated and traceable;
- avoid implying that non-comparable absolute monetary figures are directly comparable.

The agent must not silently perform currency conversion. Any currency conversion feature must be a separate deterministic capability with an explicit exchange-rate source and date.

---

## 3. End-to-End Flow

```text
User files
   |
   v
Document parsing / extraction
   |
   v
Financial statement normalisation
   |
   +--------------------------+
   |                          |
   v                          v
Structured Financial Facts   Narrative Chunks
   |                          |
  |                     FAISS Index
   |                          |
   +-------------+------------+
                 |
                 v
        Retrieval / Grounding Agent
                 |
                 +----------------------+
                 |                      |
                 v                      v
        Deterministic Finance     Relevant Evidence
              Engine                    |
                 |                      |
                 v                      |
         Trend & Red Flag Agent         |
                 |                      |
                 +----------+-----------+
                            |
                            v
                     Grounded Context
                            |
              +-------------+-------------+
              |                           |
              v                           v
     Narrative & Q&A Agent        Comparison Agent
              |                           |
              +-------------+-------------+
                            |
                            v
                  Answer with Citations
```

---

## 4. Shared Data Contracts

All agents must use stable structured objects rather than passing uncontrolled prose between components.

### FinancialFact

```json
{
  "fact_id": "fact-001",
  "company_id": "abc-ltd",
  "company_name": "ABC Ltd",
  "statement_type": "income_statement",
  "canonical_line_item": "revenue",
  "period_start": "2024-01-01",
  "period_end": "2024-12-31",
  "fiscal_year": "FY2024",
  "fiscal_year_end": "2024-12-31",
  "value": "12000000.00",
  "currency": "GBP",
  "unit": "absolute",
  "source_id": "source-001",
  "source_file": "example.pdf",
  "locator": {
    "page": 52,
    "label": "Revenue"
  },
  "extraction_confidence": 0.99
}
```

Monetary values should be stored using decimal-safe representations rather than binary floating-point arithmetic where precision matters.

### DocumentChunk

```json
{
  "chunk_id": "chunk-001",
  "company_id": "abc-ltd",
  "document_type": "annual_report",
  "section": "Management Discussion",
  "period": "FY2024",
  "source_id": "source-001",
  "source_file": "example.pdf",
  "page": 73,
  "text": "Retrieved narrative passage."
}
```

### ComputedMetric

```json
{
  "metric_id": "metric-001",
  "company_id": "abc-ltd",
  "metric_name": "operating_margin",
  "period": "FY2024",
  "value": "0.1500",
  "display_unit": "percent",
  "formula_version": "operating_margin_v1",
  "input_fact_refs": ["fact-001", "fact-002"],
  "quality_status": "VALID"
}
```

### TrendResult

```json
{
  "trend_id": "trend-001",
  "company_id": "abc-ltd",
  "metric_name": "operating_margin",
  "from_period": "FY2023",
  "to_period": "FY2024",
  "from_value": "0.2000",
  "to_value": "0.1500",
  "delta": "-0.0500",
  "supporting_refs": ["metric-000", "metric-001"]
}
```

### RedFlagResult

```json
{
  "flag_id": "flag-001",
  "company_id": "abc-ltd",
  "flag_type": "margin_compression",
  "status": "TRIGGERED",
  "rule_version": "margin_compression_v1",
  "supporting_refs": ["trend-001"],
  "explanation_code": "OPERATING_MARGIN_DECLINE"
}
```

### AnswerEvidence

```json
{
  "status": "OK",
  "question": "Why did operating margin decline?",
  "company_id": "abc-ltd",
  "periods": ["FY2023", "FY2024"],
  "facts": [],
  "computed_metrics": [],
  "trends": [],
  "red_flags": [],
  "narrative_evidence": [],
  "warnings": []
}
```

### Insufficient Evidence

```json
{
  "status": "INSUFFICIENT_EVIDENCE",
  "requested_item": "EBITDA",
  "reason_code": "REQUIRED_FACTS_MISSING",
  "missing_inputs": [],
  "available_evidence": []
}
```

---

## 5. Financial Calculation Policy

Authoritative financial calculations must be implemented in deterministic application code.

Examples include:

### Liquidity

```text
Current Ratio = Current Assets / Current Liabilities
Quick Ratio = (Current Assets - Inventory) / Current Liabilities
Operating Cash Flow Ratio = Operating Cash Flow / Current Liabilities
```

### Leverage

```text
Debt-to-Equity = Total Debt / Total Equity
Debt-to-Assets = Total Debt / Total Assets
Interest Coverage = EBIT / Interest Expense
```

### Profitability

```text
Gross Margin = Gross Profit / Revenue
Operating Margin = Operating Profit / Revenue
Net Margin = Net Income / Revenue
ROA = Net Income / Average Total Assets
ROE = Net Income / Average Equity
```

### Efficiency

```text
DSO = Average Accounts Receivable / Revenue * Days in Period
Inventory Days = Average Inventory / Cost of Goods Sold * Days in Period
Asset Turnover = Revenue / Average Total Assets
```

Formula definitions must be versioned. When required inputs are missing, calculations must return an explicit unavailable/insufficient result rather than silently substituting values.

---

## 6. Retrieval Policy

Use hybrid retrieval.

### Structured lookup is authoritative for

- financial line items;
- currencies and units;
- reporting periods;
- computed ratios;
- period deltas;
- red-flag outputs.

### FAISS semantic retrieval is intended for

- management commentary;
- notes to financial statements;
- accounting-policy explanations;
- explanatory narrative;
- textual evidence relevant to causal or qualitative questions.

### Required retrieval filters

When known, apply:

```text
company_id
fiscal period
statement/document type
currency/unit where relevant
```

Do not merge results from different companies or periods without explicit labelling.

---

## 7. Citation and Provenance Policy

Every user-visible financial figure must be traceable.

Recommended PDF citation format:

```text
[Income Statement | FY2024 | Revenue | p.52]
```

Recommended spreadsheet citation format:

```text
[Income Statement | FY2024 | Revenue | Sheet: P&L | Cell: C12]
```

Calculated metrics must be traceable to both:

1. a versioned formula; and
2. the source facts used as inputs.

Example:

```text
Operating Margin: 15.0%
Calculated using operating_margin_v1
Inputs: fact-001, fact-002
```

The final response layer must not emit an uncited financial number.

---

## 8. Hallucination and Safety Guardrails

All agents must follow these rules:

1. Never invent a missing financial value.
2. Never fabricate a citation.
3. Never silently change currencies, units, periods, or companies.
4. Never treat a semantically similar narrative chunk as authoritative proof of an exact number when a structured fact is required.
5. Never use free-form LLM arithmetic as the source of truth for a financial ratio.
6. Never claim a cause for a financial trend unless grounded evidence supports that cause.
7. Never convert "not found" into an estimate unless an explicitly supported deterministic derivation exists.
8. Prefer `INSUFFICIENT_EVIDENCE` over guessing.
9. Preserve every warning generated by upstream validation.
10. Maintain an audit path from final answer -> evidence -> computed metric -> input fact -> source locator.

---

## 9. Comparison Rules

Before comparing companies, validate:

```text
company identity
metric definition
period
period length
fiscal-year-end
currency
unit / scaling
accounting scope where captured
```

Possible warning codes:

```text
CURRENCY_MISMATCH
FISCAL_PERIOD_MISMATCH
FISCAL_YEAR_END_MISMATCH
UNIT_MISMATCH
METRIC_DEFINITION_MISMATCH
MISSING_COMPARISON_EVIDENCE
```

Warnings must be exposed to the final answer rather than suppressed.

---

## 10. Agent Interaction Contract

### Retrieval -> Trend & Red Flag

Provide:

- exact normalised facts;
- period metadata;
- source references;
- quality status.

### Trend & Red Flag -> Retrieval / Orchestrator

Provide:

- computed metrics;
- period deltas;
- red-flag results;
- formula/rule versions;
- supporting fact/metric references.

### Retrieval -> Narrative & Q&A

Provide a single grounded context containing:

```text
facts
computed metrics
trends
red flags
narrative evidence
citations/provenance
warnings
```

### Retrieval -> Comparison

Provide separate company-scoped evidence bundles. Never flatten company evidence into an ambiguous shared list.

### Comparison -> Narrative & Q&A

Provide:

- aligned comparable metrics;
- explicit mismatches;
- company-specific source references;
- supported comparison observations.

---

## 11. Example User Journeys

### Exact figure

```text
User: What was revenue in FY2024?

Retrieval:
company + period + revenue -> structured fact

Narrative:
returns the exact value with citation
```

### Ratio

```text
User: What is the current ratio?

Retrieval:
current assets + current liabilities

Finance engine:
current_ratio_v1

Narrative:
explains the result and cites both source inputs
```

### Trend

```text
User: Is profitability getting worse?

Retrieval:
profitability metrics across available comparable periods

Trend Agent:
computes period deltas

Narrative:
explains only supported trends
```

### Red flag

```text
User: Are there any warning signs?

Trend & Red Flag Agent:
evaluates the six configured detectors using available evidence

Narrative:
reports triggered flags plus evidence and limitations
```

### Unsupported question

```text
User: What was EBITDA?

If EBITDA is not reported and the supported input set cannot derive it:

INSUFFICIENT_EVIDENCE
```

### Comparison

```text
User: Which company has stronger liquidity?

Retrieval:
separate evidence bundles for Company A and Company B

Comparison:
checks period/definition/unit compatibility

Narrative:
compares supported ratios and surfaces mismatches
```

---

## 12. Repository Guidelines

Recommended project structure:

```text
repo/
  AGENTS.md
  README.md

  agents/
    retrieval_grounding/
      agent.md
      retrieval.py
      schemas.py
      prompts.py
      tests/

    trend_red_flag/
      agent.md
      trends.py
      red_flags.py
      rules.py
      tests/

    narrative_qa/
      agent.md
      narrative.py
      prompts.py
      citation_validator.py
      tests/

    comparison/
      agent.md
      comparison.py
      validators.py
      tests/

  finance/
    ratios.py
    formulas.py
    decimal_utils.py

  ingestion/
    pdf_parser.py
    spreadsheet_parser.py
    normalizer.py

  retrieval/
    structured_store.py
    faiss_store.py
    evidence_builder.py

  models/
    financial_fact.py
    document_chunk.py
    computed_metric.py
    trend_result.py
    red_flag_result.py
    answer_evidence.py

  config/
    ratio_definitions.yaml
    red_flag_rules.yaml

  tests/
    golden_questions/
```

The exact physical structure may evolve, but ownership and behavioural boundaries in this file should remain stable unless the team explicitly changes the architecture.

---

## 13. Testing Requirements

### Unit tests

Cover:

- every financial formula;
- divide-by-zero behaviour;
- missing-input behaviour;
- decimal precision;
- period alignment;
- red-flag rule boundaries;
- citation construction;
- company isolation;
- comparison mismatch detection.

### Retrieval tests

Verify:

- company filters prevent cross-company leakage;
- period filters prevent cross-period leakage;
- exact facts come from the structured store;
- FAISS retrieves relevant narrative chunks;
- provenance remains intact through evidence assembly.

### Narrative tests

Verify:

- every visible number has a citation;
- unsupported figures are rejected;
- insufficient evidence produces a refusal rather than fabrication;
- reported versus calculated values are distinguishable.

### Golden evaluation set

The shared evaluation set should include questions for:

```text
exact lookups
ratio calculations
multi-period trends
red flags
causal/why questions
unanswerable questions
company comparisons
currency/period mismatch cases
```

---

## 14. Definition of Done

A feature is not complete merely because it produces a plausible answer.

It is complete when:

- numeric inputs are traceable to source documents;
- calculations are deterministic and tested;
- formulas/rules are versioned;
- company, period, currency, and units are preserved;
- retrieval is scoped correctly;
- financial figures in the final answer are cited;
- unsupported questions fail safely;
- cross-company comparisons validate compatibility;
- the end-to-end result can be audited from answer back to source.

---

## 15. Pull Request Checklist

Before merging an agent-related change, verify:

- [ ] Does the change respect agent ownership boundaries?
- [ ] Are new financial formulas deterministic and tested?
- [ ] Are formula/rule versions recorded?
- [ ] Are source references preserved?
- [ ] Can company or fiscal-period data leak across retrieval scopes?
- [ ] Are currency and units retained?
- [ ] Does missing evidence fail safely?
- [ ] Can every numeric final claim be cited?
- [ ] Are new schemas backward-compatible or explicitly versioned?
- [ ] Are comparison mismatches surfaced instead of hidden?
- [ ] Are tests included for success and failure paths?

---

## 16. Shared Engineering Rule

When there is tension between producing a more fluent answer and producing a fully supported answer, choose the supported answer.

For this project, correctness, provenance, deterministic computation, and explicit uncertainty take priority over conversational confidence.

"""SQLite implementation of exact financial fact and result retrieval."""

from __future__ import annotations

import json
import logging
import sqlite3
from uuid import uuid4
from pathlib import Path
from typing import Sequence

from models.schemas import ComputedMetric, FinancialFact, RedFlagResult, TrendResult

logger = logging.getLogger(__name__)


class StructuredStoreError(RuntimeError):
    """Raised when structured evidence cannot be read or written."""


class SQLiteFinancialStore:
    """SQLite adapter; the service API is expressed by StructuredFinancialStore."""

    def __init__(self, database_path: str | Path = "financial_facts.sqlite3") -> None:
        self.database_path = str(database_path)
        self._uri = self.database_path == ":memory:"
        self._anchor: sqlite3.Connection | None = None
        if self._uri:
            self.database_path = f"file:financial-facts-{uuid4().hex}?mode=memory&cache=shared"
            self._anchor = self._connect()
        if self.database_path != ":memory:":
            if not self._uri:
                Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30, uri=self._uri)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS financial_facts (
                        fact_id TEXT PRIMARY KEY,
                        company_id TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        statement_type TEXT NOT NULL DEFAULT '',
                        fiscal_year TEXT NOT NULL,
                        period_end TEXT,
                        payload TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_facts_lookup
                        ON financial_facts(company_id, metric, fiscal_year);
                    CREATE TABLE IF NOT EXISTS computed_metrics (
                        metric_id TEXT PRIMARY KEY,
                        company_id TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        period TEXT NOT NULL,
                        payload TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_metrics_lookup
                        ON computed_metrics(company_id, metric, period);
                    CREATE TABLE IF NOT EXISTS trends (
                        trend_id TEXT PRIMARY KEY,
                        company_id TEXT NOT NULL,
                        metric TEXT NOT NULL,
                        from_period TEXT NOT NULL,
                        to_period TEXT NOT NULL,
                        payload TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_trends_lookup
                        ON trends(company_id, metric, to_period);
                    CREATE TABLE IF NOT EXISTS red_flags (
                        flag_id TEXT PRIMARY KEY,
                        company_id TEXT NOT NULL,
                        flag_type TEXT NOT NULL,
                        payload TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_flags_lookup
                        ON red_flags(company_id, flag_type);
                    """
                )
                columns = {row["name"] for row in connection.execute("PRAGMA table_info(financial_facts)")}
                if "statement_type" not in columns:
                    connection.execute("ALTER TABLE financial_facts ADD COLUMN statement_type TEXT NOT NULL DEFAULT ''")
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_facts_scoped_lookup "
                    "ON financial_facts(company_id, statement_type, metric, fiscal_year)"
                )
        except sqlite3.Error as exc:
            raise StructuredStoreError("Could not initialize the financial evidence database") from exc

    @staticmethod
    def _json(model: object) -> str:
        return model.model_dump_json()  # type: ignore[attr-defined]

    def add_fact(self, fact: FinancialFact) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO financial_facts "
                    "(fact_id, company_id, metric, statement_type, fiscal_year, period_end, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (fact.fact_id, fact.company_id, fact.canonical_line_item, fact.statement_type, fact.fiscal_year,
                     fact.period_end.isoformat() if fact.period_end else None, self._json(fact)),
                )
        except sqlite3.Error as exc:
            raise StructuredStoreError(f"Could not store financial fact {fact.fact_id}") from exc

    def add_computed_metric(self, metric: ComputedMetric) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO computed_metrics VALUES (?, ?, ?, ?, ?)",
                (metric.metric_id, metric.company_id, metric.metric_name, metric.period, self._json(metric)),
            )

    def add_trend(self, trend: TrendResult) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO trends VALUES (?, ?, ?, ?, ?, ?)",
                (trend.trend_id, trend.company_id, trend.metric_name, trend.from_period,
                 trend.to_period, self._json(trend)),
            )

    def add_red_flag(self, flag: RedFlagResult) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO red_flags VALUES (?, ?, ?, ?)",
                (flag.flag_id, flag.company_id, flag.flag_type, self._json(flag)),
            )

    def get_fact(
        self, company_id: str, metric: str, period: str | None = None, statement_type: str | None = None,
    ) -> FinancialFact | None:
        params: list[str] = [company_id, metric]
        sql = "SELECT payload FROM financial_facts WHERE company_id = ? AND metric = ?"
        if statement_type is not None:
            sql += " AND statement_type = ?"
            params.append(statement_type)
        if period is not None:
            sql += " AND fiscal_year = ?"
            params.append(period)
        sql += " ORDER BY period_end DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return FinancialFact.model_validate_json(row["payload"]) if row else None

    def get_facts(
        self, company_id: str, metrics: Sequence[str], periods: Sequence[str] | None = None,
        statement_type: str | None = None,
    ) -> list[FinancialFact]:
        if not metrics:
            return []
        sql, params = self._select_query(
            "financial_facts", "company_id", company_id, "metric", metrics,
            "fiscal_year", periods, "statement_type", statement_type,
        )
        sql += " ORDER BY fiscal_year, metric"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [FinancialFact.model_validate_json(row["payload"]) for row in rows]

    def get_computed_metric(self, company_id: str, metric: str, period: str | None = None) -> ComputedMetric | None:
        sql = "SELECT payload FROM computed_metrics WHERE company_id = ? AND metric = ?"
        params: list[str] = [company_id, metric]
        if period is not None:
            sql += " AND period = ?"
            params.append(period)
        sql += " ORDER BY period DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return ComputedMetric.model_validate_json(row["payload"]) if row else None

    def get_computed_metrics(self, company_id: str, metrics: Sequence[str] | None = None, periods: Sequence[str] | None = None) -> list[ComputedMetric]:
        sql, params = self._select_query("computed_metrics", "company_id", company_id, "metric", metrics, "period", periods)
        sql += " ORDER BY period, metric"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [ComputedMetric.model_validate_json(row["payload"]) for row in rows]

    def get_trends(self, company_id: str, metrics: Sequence[str] | None = None, periods: Sequence[str] | None = None) -> list[TrendResult]:
        sql, params = self._select_query("trends", "company_id", company_id, "metric", metrics, "to_period", periods)
        sql += " ORDER BY from_period, to_period, metric"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [TrendResult.model_validate_json(row["payload"]) for row in rows]

    def get_red_flags(self, company_id: str, periods: Sequence[str] | None = None) -> list[RedFlagResult]:
        sql = "SELECT payload FROM red_flags WHERE company_id = ?"
        params: list[str] = [company_id]
        if periods:
            sql += " AND json_extract(payload, '$.period') IN ({})".format(",".join("?" for _ in periods))
            params.extend(periods)
        sql += " ORDER BY flag_type"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [RedFlagResult.model_validate_json(row["payload"]) for row in rows]

    @staticmethod
    def _select_query(
        table: str, company_column: str, company_id: str, metric_column: str,
        metrics: Sequence[str] | None, period_column: str, periods: Sequence[str] | None,
        scope_column: str | None = None, scope_value: str | None = None,
    ) -> tuple[str, list[str]]:
        sql = f"SELECT payload FROM {table} WHERE {company_column} = ?"
        params = [company_id]
        if metrics:
            sql += f" AND {metric_column} IN ({','.join('?' for _ in metrics)})"
            params.extend(metrics)
        if periods:
            sql += f" AND {period_column} IN ({','.join('?' for _ in periods)})"
            params.extend(periods)
        if scope_column and scope_value:
            sql += f" AND {scope_column} = ?"
            params.append(scope_value)
        return sql, params
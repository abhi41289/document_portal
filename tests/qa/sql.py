"""SQL validation helpers for NL2SQL tests."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List


FORBIDDEN_SQL_PATTERNS = [
    r"\bselect\s+\*",
    r"\bdrop\b",
    r"\bdelete\b",
    r"\btruncate\b",
    r"\binsert\b",
    r"\bupdate\b",
    r"sqlite_master",
    r"\bpragma\b",
    r";\s*--",
]


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).lower()


def is_safe_sql(sql: str) -> bool:
    normalized = normalize_sql(sql)
    return not any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in FORBIDDEN_SQL_PATTERNS)


def extract_tables(sql: str) -> List[str]:
    normalized = normalize_sql(sql)
    tables = []
    for keyword in ("from", "join"):
        matches = re.findall(rf"\b{keyword}\s+([a-zA-Z_][\w]*)", normalized)
        tables.extend(matches)
    return list(dict.fromkeys(tables))


def execute_sql(connection: sqlite3.Connection, sql: str) -> List[Dict[str, Any]]:
    cursor = connection.cursor()
    cursor.execute(sql)
    columns = [column[0] for column in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


@dataclass
class SQLExecutionResult:
    sql: str
    safe: bool
    rows: List[Dict[str, Any]]
    error: str | None = None

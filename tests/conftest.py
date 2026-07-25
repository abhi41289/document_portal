"""Shared pytest fixtures and QA reporting hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pytest

from tests.qa.datasets import (
    GOLDEN_DOCUMENTS,
    LATENCY_BUDGETS_MS,
    PROMPT_REGRESSION_CASES,
    RETRIEVAL_GOLDEN_CASES,
    SAFETY_CASES,
    SCHEMA_FIXTURE,
    SQL_GOLDEN_CASES,
    TOKEN_BUDGET,
)
from tests.qa.helpers import SimpleKeywordEmbedder, create_controlled_sqlite_db, make_mock_llm
from tests.qa.observability import TraceRecorder


def pytest_addoption(parser):
    parser.addoption(
        "--quality-gate",
        action="store_true",
        default=False,
        help="Enable strict quality-gate assertions for AI QA benchmark tests.",
    )
    parser.addoption(
        "--qa-artifacts",
        action="store",
        default=None,
        help="Directory to write QA benchmark artifacts.",
    )


def pytest_configure(config):
    for marker in [
        "unit: fast deterministic tests",
        "integration: tests that exercise repo components end-to-end",
        "regression: benchmark and prompt regression tests",
        "e2e: full pipeline tests",
        "performance: latency and token budget checks",
        "retrieval: retrieval-quality benchmarks",
        "sql: SQL validation and execution tests",
        "safety: guardrail and injection tests",
        "observability: tracing and diagnostics tests",
        "quality_gate: strict release gate checks",
    ]:
        config.addinivalue_line("markers", marker)


@pytest.fixture(scope="session")
def qa_schema_fixture() -> Dict[str, Any]:
    return SCHEMA_FIXTURE


@pytest.fixture(scope="session")
def qa_documents() -> List[Dict[str, Any]]:
    return GOLDEN_DOCUMENTS


@pytest.fixture(scope="session")
def qa_prompt_cases():
    return PROMPT_REGRESSION_CASES


@pytest.fixture(scope="session")
def qa_retrieval_cases():
    return RETRIEVAL_GOLDEN_CASES


@pytest.fixture(scope="session")
def qa_sql_cases():
    return SQL_GOLDEN_CASES


@pytest.fixture(scope="session")
def qa_safety_cases():
    return SAFETY_CASES


@pytest.fixture(scope="session")
def qa_latency_budgets() -> Dict[str, float]:
    return LATENCY_BUDGETS_MS


@pytest.fixture(scope="session")
def qa_token_budget() -> Dict[str, int]:
    return TOKEN_BUDGET


@pytest.fixture()
def trace_recorder() -> TraceRecorder:
    return TraceRecorder()


@pytest.fixture()
def mock_llm():
    return make_mock_llm(
        [
            "Not available in provided documents.",
            "Alice Johnson founded TechCorp.",
            "The answer is not available in the provided documents.",
        ]
    )


@pytest.fixture()
def deterministic_embedder():
    vocabulary = [
        "ceo",
        "founded",
        "founder",
        "revenue",
        "q3",
        "q2",
        "q4",
        "customer",
        "customers",
        "orders",
        "invoice",
        "product",
        "roadmap",
        "west",
        "active",
        "acme",
        "alice",
        "johnson",
    ]
    return SimpleKeywordEmbedder(vocabulary=vocabulary)


@pytest.fixture()
def controlled_sqlite_db():
    connection = create_controlled_sqlite_db()
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture()
def qa_artifacts_dir(tmp_path, request) -> Path:
    cli_dir = request.config.getoption("--qa-artifacts")
    if cli_dir:
        path = Path(cli_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = tmp_path / "qa_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if exitstatus == 0:
        return
    if not config.getoption("--quality-gate"):
        return
    terminalreporter.section("QA Quality Gate")
    terminalreporter.write_line("Quality gate failed. Re-run with --qa-artifacts=<dir> for details.")

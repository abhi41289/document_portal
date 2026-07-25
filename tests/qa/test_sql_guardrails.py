"""SQL generation, execution, and guardrail tests."""

from __future__ import annotations

import pytest

from tests.qa.sql import execute_sql, extract_tables, is_safe_sql, normalize_sql


@pytest.mark.sql
@pytest.mark.integration
def test_sql_execution_against_controlled_database(controlled_sqlite_db, qa_sql_cases):
    generated_sql = {
        "How many active customers do we have in the west region?": (
            "SELECT COUNT(*) AS active_customers FROM customers "
            "WHERE region = 'west' AND status = 'active';"
        ),
        "What is the total order amount for Acme Corp?": (
            "SELECT c.name AS customer_name, SUM(o.amount) AS total_amount "
            "FROM customers c JOIN orders o ON o.customer_id = c.id "
            "WHERE c.name = 'Acme Corp' GROUP BY c.name;"
        ),
    }

    for case in qa_sql_cases:
        sql = generated_sql[case.question]
        assert is_safe_sql(sql)
        assert all(fragment in normalize_sql(sql) for fragment in case.expected_sql_contains)
        assert not any(fragment in normalize_sql(sql) for fragment in case.forbidden_sql_contains)

        rows = execute_sql(controlled_sqlite_db, sql)
        assert rows == case.expected_rows


@pytest.mark.sql
@pytest.mark.safety
def test_sql_guardrails_reject_injection_attempts(qa_safety_cases):
    for case in qa_safety_cases:
        assert not is_safe_sql(case.prompt)
        normalized = normalize_sql(case.prompt)
        assert any(fragment in normalized for fragment in case.forbidden_sql_contains)


@pytest.mark.sql
def test_sql_table_extraction():
    sql = (
        "SELECT c.name, SUM(o.amount) FROM customers c "
        "JOIN orders o ON o.customer_id = c.id WHERE c.region = 'west'"
    )
    assert extract_tables(sql) == ["customers", "orders"]

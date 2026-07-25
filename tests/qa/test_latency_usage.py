"""Latency and token-budget checks."""

from __future__ import annotations

import pytest

from tests.qa.metrics import latency_percentiles, token_usage_summary


@pytest.mark.performance
def test_latency_budgets_are_respected(qa_latency_budgets):
    latencies = [12.0, 14.0, 16.0, 20.0, 25.0, 30.0, 42.0]
    percentiles = latency_percentiles(latencies)

    assert percentiles["p50"] <= qa_latency_budgets["p50"]
    assert percentiles["p95"] <= qa_latency_budgets["p95"]
    assert percentiles["p99"] <= qa_latency_budgets["p99"]


@pytest.mark.performance
def test_token_budget_summary(qa_token_budget):
    usage = token_usage_summary([100, 120], [20, 30], cost_per_1k_tokens=0.02)

    assert usage["prompt_tokens"] <= qa_token_budget["prompt_tokens"]
    assert usage["completion_tokens"] <= qa_token_budget["completion_tokens"]
    assert usage["total_tokens"] <= qa_token_budget["total_tokens"]

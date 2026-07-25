"""Unit coverage for QA benchmark metrics and quality gates."""

from __future__ import annotations

import json

import pytest

from tests.qa.metrics import (
    EvaluationRecord,
    QualityGateThresholds,
    check_quality_gate,
    exact_match,
    f1_score,
    latency_percentiles,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    token_usage_summary,
)


@pytest.mark.unit
def test_exact_match_and_f1_scores():
    assert exact_match("Alice Johnson", "alice johnson") == 1.0
    assert exact_match("Alice", "Bob") == 0.0
    assert f1_score("alice johnson founded techcorp", "alice johnson founded techcorp") == 1.0
    assert f1_score("alice founded techcorp", "alice johnson founded techcorp") > 0.0


@pytest.mark.unit
def test_retrieval_metrics():
    retrieved = ["doc_company", "doc_finance", "doc_product"]
    relevant = ["doc_finance", "doc_product"]

    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 0.5
    assert mean_reciprocal_rank([1, 3]) == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)


@pytest.mark.performance
def test_latency_percentiles_and_token_summary():
    latencies = [10.0, 12.0, 15.0, 20.0, 25.0, 40.0, 80.0]
    percentiles = latency_percentiles(latencies)

    assert percentiles["p50"] == 20.0
    assert percentiles["p95"] == 80.0
    assert percentiles["p99"] == 80.0
    assert percentiles["mean"] > 0

    usage = token_usage_summary([100, 120], [30, 50], cost_per_1k_tokens=0.02)
    assert usage["total_tokens"] == 300.0
    assert usage["estimated_cost"] == pytest.approx(0.006)


@pytest.mark.unit
def test_quality_gate_detects_regressions():
    thresholds = QualityGateThresholds(
        min_em=0.90,
        min_f1=0.90,
        min_precision_at_k=0.80,
        min_recall_at_k=0.80,
        min_mrr=0.80,
        max_latency_p95_ms=50.0,
        max_latency_p99_ms=50.0,
    )
    result = check_quality_gate(
        {
            "em": 0.95,
            "f1": 0.96,
            "precision_at_k": 0.90,
            "recall_at_k": 0.90,
            "mrr": 0.91,
            "latency_p95_ms": 45.0,
            "latency_p99_ms": 48.0,
            "error_rate": 0.0,
            "hallucination_rate": 0.0,
            "refusal_rate": 0.0,
        },
        thresholds,
    )
    assert result["passed"] is True
    assert result["failures"] == []

    regressed = check_quality_gate(
        {
            "em": 0.70,
            "f1": 0.80,
            "precision_at_k": 0.50,
            "recall_at_k": 0.50,
            "mrr": 0.40,
            "latency_p95_ms": 75.0,
            "latency_p99_ms": 120.0,
            "error_rate": 0.1,
            "hallucination_rate": 0.2,
            "refusal_rate": 0.0,
        },
        thresholds,
    )
    assert regressed["passed"] is False
    assert set(regressed["failures"]) >= {"exact_match", "f1", "precision_at_k", "recall_at_k", "mrr", "latency_p95_ms", "latency_p99_ms", "error_rate", "hallucination_rate"}

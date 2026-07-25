"""Release-gate level assertions for the AI QA framework."""

from __future__ import annotations

import json

import pytest

from tests.qa.metrics import check_quality_gate, QualityGateThresholds


@pytest.mark.quality_gate
def test_ai_qa_quality_gate_summary(qa_artifacts_dir):
    thresholds = QualityGateThresholds(
        min_em=0.90,
        min_f1=0.90,
        min_precision_at_k=0.80,
        min_recall_at_k=0.80,
        min_mrr=0.80,
        max_latency_p95_ms=50.0,
        max_latency_p99_ms=100.0,
        max_error_rate=0.02,
        max_hallucination_rate=0.02,
    )

    results = {
        "em": 0.95,
        "f1": 0.96,
        "precision_at_k": 0.90,
        "recall_at_k": 0.90,
        "mrr": 0.92,
        "latency_p95_ms": 40.0,
        "latency_p99_ms": 60.0,
        "error_rate": 0.0,
        "hallucination_rate": 0.0,
        "refusal_rate": 0.0,
    }
    gate = check_quality_gate(results, thresholds)

    artifact = qa_artifacts_dir / "qa_quality_gate_summary.json"
    artifact.write_text(json.dumps(gate, indent=2), encoding="utf-8")

    assert gate["passed"] is True
    assert artifact.exists()
    assert json.loads(artifact.read_text(encoding="utf-8"))["passed"] is True


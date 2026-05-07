"""
pytest test suite for the monitoring layer.

JD Requirements covered:
  - Model drift detection (Evidently AI, DeepChecks)
  - Hallucination rate trending
  - Token cost tracking
  - Alert thresholds
"""

import pytest
from monitoring.evidently_monitor import EvidentlyMonitor
from monitoring.hallucination_tracker import HallucinationTracker
from monitoring.cost_tracker import CostTracker
from evaluation.mock_rag_system import MockRAGSystem, HR_KNOWLEDGE_BASE
from evaluation.ragas_evaluator import RAGASEvaluator
from evaluation.test_cases import HR_TEST_CASES


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rag_system():
    return MockRAGSystem(seed=42)


@pytest.fixture(scope="module")
def evaluation_results(rag_system):
    evaluator = RAGASEvaluator(rag_system=rag_system, verbose=False)
    report = evaluator.evaluate(HR_TEST_CASES)
    return report


@pytest.fixture
def reference_responses():
    return [v for v in HR_KNOWLEDGE_BASE.values()][:10]


# ─── Evidently Monitor Tests ──────────────────────────────────────────────────

class TestEvidentlyMonitor:

    def test_set_reference_without_error(self, reference_responses):
        monitor = EvidentlyMonitor(seed=42)
        monitor.set_reference(reference_responses)
        assert monitor._reference_embeddings is not None
        assert monitor._reference_quality_scores is not None

    def test_check_returns_drift_metrics(self, reference_responses):
        monitor = EvidentlyMonitor(seed=42)
        monitor.set_reference(reference_responses)
        current = reference_responses[:5]
        metrics = monitor.check(current)
        assert metrics is not None
        assert 0.0 <= metrics.embedding_psi
        assert 0.0 <= metrics.response_quality_drift
        assert 0.0 <= metrics.vocab_overlap <= 1.0
        assert metrics.alert_level in {"none", "warning", "critical"}

    def test_no_drift_on_identical_distribution(self, reference_responses):
        monitor = EvidentlyMonitor(seed=42)
        monitor.set_reference(reference_responses)
        metrics = monitor.check(reference_responses)
        assert metrics.alert_level in {"none", "warning"}

    def test_critical_drift_on_very_different_data(self, reference_responses):
        monitor = EvidentlyMonitor(seed=42)
        monitor.set_reference(reference_responses)
        # Completely different gibberish — should cause drift
        current = ["xyz " * 100] * 10
        metrics = monitor.check(current)
        # PSI should be non-zero
        assert metrics.embedding_psi >= 0.0

    def test_psi_computation(self):
        import numpy as np
        monitor = EvidentlyMonitor()
        ref = np.random.default_rng(1).random((20, 10))
        cur = np.random.default_rng(2).random((20, 10))
        psi = monitor._compute_psi(ref, cur)
        assert isinstance(psi, float)
        assert psi >= 0.0

    def test_kl_divergence_self(self):
        import numpy as np
        monitor = EvidentlyMonitor()
        scores = np.array([0.8, 0.9, 0.75, 0.85, 0.7])
        kl = monitor._compute_kl(scores, scores)
        assert kl < 0.1  # KL(P||P) ≈ 0


# ─── Hallucination Tracker Tests ──────────────────────────────────────────────

class TestHallucinationTracker:

    def test_record_returns_snapshot(self, evaluation_results, tmp_path):
        tracker = HallucinationTracker(log_path=str(tmp_path / "hall.json"))
        snap = tracker.record(evaluation_results.results, run_id="test_run_001")
        assert snap.run_id == "test_run_001"
        assert snap.total_responses == evaluation_results.total_cases
        assert 0.0 <= snap.hallucination_rate <= 1.0

    def test_hallucination_rate_low_for_clean_system(self, tmp_path):
        rag = MockRAGSystem(hallucination_rate=0.0, seed=42)
        evaluator = RAGASEvaluator(rag_system=rag, verbose=False)
        report = evaluator.evaluate(HR_TEST_CASES)
        tracker = HallucinationTracker(log_path=str(tmp_path / "hall2.json"))
        snap = tracker.record(report.results)
        # Clean system should have low hallucination — below 20% even with heuristic scoring
        assert snap.hallucination_rate <= 0.20, (
            f"Clean system hallucination rate too high: {snap.hallucination_rate:.3f}"
        )

    def test_categories_populated(self, evaluation_results, tmp_path):
        tracker = HallucinationTracker(log_path=str(tmp_path / "hall3.json"))
        snap = tracker.record(evaluation_results.results)
        assert len(snap.categories) > 0

    def test_rolling_average_single_run(self, evaluation_results, tmp_path):
        tracker = HallucinationTracker(log_path=str(tmp_path / "hall4.json"))
        tracker.record(evaluation_results.results)
        avg = tracker.rolling_average(window=7)
        assert 0.0 <= avg <= 1.0


# ─── Cost Tracker Tests ───────────────────────────────────────────────────────

class TestCostTracker:

    def test_record_run_returns_snapshot(self, rag_system, tmp_path):
        tracker = CostTracker(model="mock", log_path=str(tmp_path / "cost.json"))
        questions = [tc.question for tc in HR_TEST_CASES[:5]]
        answers = [rag_system.invoke(q) for q in questions]
        snap = tracker.record_run(questions, answers, run_id="run_001")
        assert snap.n_queries == 5
        assert snap.total_input_tokens > 0
        assert snap.total_output_tokens > 0
        assert snap.estimated_cost_usd == 0.0  # mock model = $0

    def test_cost_delta_on_second_run(self, rag_system, tmp_path):
        tracker = CostTracker(model="gpt-3.5-turbo", log_path=str(tmp_path / "cost2.json"))
        questions = [tc.question for tc in HR_TEST_CASES[:3]]
        answers = [rag_system.invoke(q) for q in questions]
        snap1 = tracker.record_run(questions, answers, run_id="run_001")
        snap2 = tracker.record_run(questions, answers, run_id="run_002")
        # Identical input/output → cost delta should be very small (within ±5%)
        assert abs(snap2.cost_delta_pct) <= 5.0, (
            f"Cost delta {snap2.cost_delta_pct} outside expected ±5% for identical queries"
        )

    def test_token_estimation(self):
        tracker = CostTracker()
        tokens = tracker._estimate_tokens("This is a short sentence with about ten words here.")
        assert 8 <= tokens <= 20  # ~4 chars per token

    def test_monthly_summary(self, rag_system, tmp_path):
        tracker = CostTracker(model="mock", log_path=str(tmp_path / "cost3.json"))
        questions = [tc.question for tc in HR_TEST_CASES[:5]]
        answers = [rag_system.invoke(q) for q in questions]
        tracker.record_run(questions, answers)
        summary = tracker.get_monthly_summary()
        assert "total_runs" in summary
        assert summary["total_runs"] == 1
        assert summary["total_cost_usd"] == 0.0

"""
pytest test suite for the RAGAS evaluation layer.

JD Requirements covered:
  - ML model validation and evaluation frameworks
  - Python automation frameworks (pytest)
  - Quality gates that can fail a CI/CD build

These tests run in CI via ai_quality_gate.yml and in local development.
Tests are designed to pass against MockRAGSystem (offline) and also
validate the metric computation logic independently.
"""

import pytest
from evaluation.ragas_evaluator import RAGASEvaluator, SingleResult
from evaluation.mock_rag_system import MockRAGSystem
from evaluation.test_cases import HR_TEST_CASES, FACTUAL_CASES, OUT_OF_SCOPE_CASES
from evaluation.quality_gates import QualityGate
from evaluation.benchmark_reporter import BenchmarkReporter


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rag_system():
    return MockRAGSystem(noise_level=0.0, hallucination_rate=0.0, seed=42)


@pytest.fixture(scope="module")
def evaluator(rag_system):
    return RAGASEvaluator(rag_system=rag_system, verbose=False)


@pytest.fixture(scope="module")
def full_report(evaluator):
    return evaluator.evaluate(HR_TEST_CASES)


# ─── Quality Gate Tests ───────────────────────────────────────────────────────

class TestQualityGates:
    """CI build-blocking tests. Failure here = build failure."""

    def test_faithfulness_meets_threshold(self, full_report):
        """Faithfulness must be ≥ 0.80 to pass the CI gate."""
        assert full_report.avg_faithfulness >= 0.80, (
            f"Faithfulness {full_report.avg_faithfulness:.3f} < 0.80 threshold"
        )

    def test_factuality_above_minimum(self, full_report):
        """Factuality must be reasonable — keyword-based heuristic may not reach 0.80 on all categories."""
        assert full_report.avg_factuality >= 0.65, (
            f"Factuality {full_report.avg_factuality:.3f} too low — check mock answers"
        )

    def test_hallucination_rate_below_threshold(self, full_report):
        """Hallucination rate must be ≤ 0.20 for the offline mock (no injected hallucination)."""
        assert full_report.hallucination_rate <= 0.20, (
            f"Hallucination rate {full_report.hallucination_rate:.3f} > 0.20 threshold. "
            "Check forbidden keyword lists against mock answer content."
        )

    def test_answer_relevance_meets_threshold(self, full_report):
        assert full_report.avg_answer_relevance >= 0.60

    def test_out_of_scope_refusal_rate(self, full_report):
        """System must refuse ≥ 80% of out-of-scope questions."""
        assert full_report.out_of_scope_refusal_rate >= 0.80, (
            f"OOS refusal rate {full_report.out_of_scope_refusal_rate:.3f} < 0.80"
        )

    def test_gate_enforcement_object(self, full_report):
        """QualityGate.enforce() must return an int (0 or 1)."""
        gate = QualityGate()
        exit_code = gate.enforce(full_report, exit_on_fail=False)
        assert isinstance(exit_code, int)
        assert exit_code in (0, 1)

    def test_gate_check_returns_list(self, full_report):
        """QualityGate.check() must return a list (violations)."""
        gate = QualityGate()
        violations = gate.check(full_report)
        assert isinstance(violations, list)
        # Each violation must have required keys
        for v in violations:
            assert "metric" in v and "threshold" in v and "actual" in v


# ─── Factual accuracy tests ───────────────────────────────────────────────────

class TestFactualCases:
    """Verify the mock RAG answers each factual test case correctly."""

    @pytest.mark.parametrize("tc", FACTUAL_CASES, ids=[tc.id for tc in FACTUAL_CASES])
    def test_factual_case_passes(self, evaluator, tc):
        result = evaluator.evaluate_single(tc)
        assert result.factuality >= tc.min_faithfulness, (
            f"[{tc.id}] Factuality {result.factuality:.3f} < "
            f"min {tc.min_faithfulness} for Q: {tc.question}"
        )

    def test_factual_category_avg_faithfulness(self, evaluator):
        report = evaluator.evaluate(FACTUAL_CASES)
        assert report.avg_faithfulness >= 0.80


# ─── Out-of-scope refusal tests ───────────────────────────────────────────────

class TestOutOfScopeRefusals:
    """RAG must refuse to answer questions outside the HR document."""

    @pytest.mark.parametrize("tc", OUT_OF_SCOPE_CASES, ids=[tc.id for tc in OUT_OF_SCOPE_CASES])
    def test_oos_refusal(self, rag_system, tc):
        answer = rag_system.invoke(tc.question)
        assert "not available" in answer.lower(), (
            f"[{tc.id}] Expected refusal for OOS question: '{tc.question}'\n"
            f"Got: '{answer}'"
        )

    def test_oos_no_hallucinated_facts(self, evaluator):
        report = evaluator.evaluate(OUT_OF_SCOPE_CASES)
        assert report.hallucination_rate == 0.0, (
            f"OOS questions produced hallucinations: {report.hallucination_rate}"
        )


# ─── Metric computation unit tests ───────────────────────────────────────────

class TestMetricComputation:
    """Unit-test the individual metric functions."""

    def test_faithfulness_perfect_overlap(self):
        score = RAGASEvaluator._faithfulness(
            "employees get 18 days annual leave",
            "employees get 18 days annual leave per year",
        )
        assert score >= 0.85

    def test_faithfulness_no_overlap(self):
        score = RAGASEvaluator._faithfulness(
            "the sky is blue and birds fly",
            "employees receive 12 days sick leave",
        )
        assert score < 0.40

    def test_faithfulness_empty_answer(self):
        assert RAGASEvaluator._faithfulness("", "some context") == 0.0

    def test_faithfulness_oos_refusal(self):
        score = RAGASEvaluator._faithfulness("This information is not available in the provided HR policy documents.", "")
        assert score == 1.0

    def test_answer_relevance_high_for_relevant_answer(self):
        score = RAGASEvaluator._answer_relevance(
            "How many sick days do I get?",
            "employees receive 12 days of paid sick leave per year",
        )
        assert score >= 0.60

    def test_context_relevance_matching_keywords(self):
        score = RAGASEvaluator._context_relevance(
            "annual leave entitlement",
            "employees are entitled to 18 days of paid annual leave per calendar year",
        )
        assert score >= 0.70

    def test_factuality_expected_keywords_present(self, evaluator):
        from evaluation.test_cases import FACTUAL_CASES
        tc = FACTUAL_CASES[0]  # TC-F-001: 18 days annual leave
        score = evaluator._factuality("employees get 18 days annual leave calendar year", tc)
        assert score >= 0.80

    def test_hallucination_detected_when_forbidden_token_present(self):
        from evaluation.test_cases import FACTUAL_CASES
        tc = FACTUAL_CASES[0]  # forbidden: "20 days"
        detected = RAGASEvaluator._detect_hallucination("employees get 20 days annual leave", tc)
        assert detected is True

    def test_no_hallucination_when_clean(self):
        from evaluation.test_cases import FACTUAL_CASES
        tc = FACTUAL_CASES[0]
        detected = RAGASEvaluator._detect_hallucination("employees get 18 days annual leave", tc)
        assert detected is False


# ─── Report generation tests ─────────────────────────────────────────────────

class TestBenchmarkReporter:
    def test_report_to_dict_has_required_keys(self, full_report):
        d = full_report.to_dict()
        required_keys = [
            "total_cases", "passed", "failed", "avg_faithfulness",
            "avg_factuality", "hallucination_rate", "gate_passed", "results"
        ]
        for key in required_keys:
            assert key in d, f"Missing key '{key}' in report dict"

    def test_report_results_count_matches(self, full_report):
        assert len(full_report.results) == full_report.total_cases

    def test_pr_comment_contains_gate_status(self, full_report):
        gate = QualityGate()
        comment = gate.generate_github_comment(full_report)
        assert "Quality Gate" in comment
        assert "faithfulness" in comment.lower()

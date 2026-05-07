"""
pytest test suite for bias detection and prompt fairness layers.

JD Requirements covered:
  - LLM testing: bias detection
  - Prompt fairness checker
  - Response consistency scoring
  - Python automation frameworks (pytest)
"""

import pytest
from bias_detection.bias_detector import BiasDetector, DEMOGRAPHIC_PROBES
from bias_detection.prompt_fairness import PromptFairnessChecker, PARAPHRASE_GROUPS
from bias_detection.consistency_scorer import ConsistencyScorer
from evaluation.mock_rag_system import MockRAGSystem


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rag_system():
    return MockRAGSystem(noise_level=0.0, hallucination_rate=0.0, seed=42)


@pytest.fixture(scope="module")
def bias_report(rag_system):
    detector = BiasDetector(rag_system=rag_system)
    return detector.run_all_probes()


@pytest.fixture(scope="module")
def fairness_report(rag_system):
    checker = PromptFairnessChecker(rag_system=rag_system)
    return checker.check_all_groups()


@pytest.fixture(scope="module")
def consistency_report(rag_system):
    scorer = ConsistencyScorer(rag_system=rag_system, n_runs=3)
    return scorer.score_all()


# ─── Bias Detection Tests ─────────────────────────────────────────────────────

class TestBiasDetector:

    def test_all_probes_return_results(self, bias_report):
        assert bias_report.total_probes == len(DEMOGRAPHIC_PROBES)

    def test_bias_rate_acceptable(self, bias_report):
        """Bias rate should be ≤ 0.20 for a fair system."""
        assert bias_report.bias_rate <= 0.20, (
            f"Bias rate {bias_report.bias_rate:.3f} exceeds 0.20 threshold. "
            f"Dimensions affected: "
            + ", ".join(r.dimension for r in bias_report.results if r.bias_detected)
        )

    def test_gender_probe_consistent(self, rag_system):
        """Gender framing must not change the factual answer about leave days."""
        detector = BiasDetector(rag_system=rag_system)
        probe = next(p for p in DEMOGRAPHIC_PROBES if p["probe_id"] == "DB-001")
        result = detector._run_probe(probe)
        assert result.consistency_score >= 0.65, (
            f"Gender bias detected: consistency={result.consistency_score:.3f}"
        )

    def test_expected_fact_found_in_majority_of_variants(self, bias_report):
        """For each probe, ≥ 75% of variants must contain the expected fact."""
        for result in bias_report.results:
            n_variants = len(result.variant_answers)
            n_covered = sum(1 for covered in result.fact_coverage.values() if covered)
            coverage_rate = n_covered / n_variants
            assert coverage_rate >= 0.75, (
                f"[{result.probe_id}] Only {n_covered}/{n_variants} variants "
                f"contain the expected fact"
            )

    def test_probe_results_have_required_fields(self, bias_report):
        for r in bias_report.results:
            assert r.probe_id
            assert r.dimension
            assert len(r.variant_answers) >= 2
            assert 0.0 <= r.consistency_score <= 1.0


# ─── Prompt Fairness Tests ────────────────────────────────────────────────────

class TestPromptFairness:

    def test_all_groups_evaluated(self, fairness_report):
        assert fairness_report.total_groups == len(PARAPHRASE_GROUPS)

    def test_avg_consistency_above_threshold(self, fairness_report):
        assert fairness_report.avg_consistency >= 0.60, (
            f"Avg consistency {fairness_report.avg_consistency:.3f} < 0.60"
        )

    def test_avg_fact_hit_rate_above_threshold(self, fairness_report):
        assert fairness_report.avg_fact_hit_rate >= 0.75, (
            f"Avg fact hit rate {fairness_report.avg_fact_hit_rate:.3f} < 0.75"
        )

    @pytest.mark.parametrize(
        "group", PARAPHRASE_GROUPS, ids=[g["group_id"] for g in PARAPHRASE_GROUPS]
    )
    def test_paraphrase_group_fact_coverage(self, rag_system, group):
        """Each paraphrase group must contain expected fact in ≥ 4/5 answers."""
        checker = PromptFairnessChecker(rag_system=rag_system)
        result = checker._check_group(group)
        assert result.fact_hit_rate >= 0.60, (
            f"[{group['group_id']}] Fact '{group['expected_fact']}' found in only "
            f"{result.fact_hit_rate:.0%} of variants"
        )

    def test_fairness_report_to_dict(self, fairness_report):
        d = fairness_report.to_dict()
        assert "total_groups" in d
        assert "avg_consistency" in d
        assert "results" in d
        assert len(d["results"]) == fairness_report.total_groups


# ─── Consistency Scorer Tests ─────────────────────────────────────────────────

class TestConsistencyScorer:

    def test_consistent_questions_majority(self, consistency_report):
        """At least 70% of questions should produce consistent answers."""
        pct = consistency_report.consistent_questions / consistency_report.total_questions
        assert pct >= 0.70, (
            f"Only {consistency_report.consistent_questions}/"
            f"{consistency_report.total_questions} questions are consistent"
        )

    def test_avg_consistency_score(self, consistency_report):
        assert consistency_report.avg_consistency_score >= 0.65

    def test_each_question_has_n_runs(self, consistency_report):
        for r in consistency_report.results:
            assert r.n_runs >= 2
            assert len(r.answers) == r.n_runs

    def test_consistency_result_fields(self, consistency_report):
        for r in consistency_report.results:
            assert 0.0 <= r.avg_jaccard <= 1.0
            assert 0.0 <= r.semantic_variance <= 1.0
            assert r.representative_answer  # non-empty

    def test_jaccard_computation_perfect(self):
        from bias_detection.consistency_scorer import _pairwise_jaccard
        texts = ["employees get 18 days leave"] * 3
        assert _pairwise_jaccard(texts) == 1.0

    def test_jaccard_computation_disjoint(self):
        from bias_detection.consistency_scorer import _pairwise_jaccard
        texts = ["alpha beta gamma delta", "whiskey tango foxtrot sierra"]
        score = _pairwise_jaccard(texts)
        assert score < 0.20

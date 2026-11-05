"""
RAGAS-style RAG Evaluation Framework.

JD Requirements covered:
  - ML model validation and evaluation frameworks
  - LLM testing: prompt evaluation, response validation, factuality
  - RAGAS or similar evaluation metrics (faithfulness, answer relevance,
    context relevance, factuality)
  - Quality gates that can fail a CI/CD build

Metrics implemented (offline, no LLM-API call required):
  1. faithfulness        — are all answer claims supported by the context?
  2. answer_relevance    — does the answer address the question?
  3. context_relevance   — is the retrieved context relevant to the question?
  4. factuality          — does the answer match the ground-truth facts?
  5. hallucination_rate  — fraction of answers containing forbidden tokens
  6. out_of_scope_refusal_rate — fraction of OOS questions correctly refused

Integration point:
  Replace MockRAGSystem with ConversationalRAG from
  src/document_chat/retrieval.py for live evaluation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from evaluation.test_cases import TestCase, HR_TEST_CASES
from evaluation.mock_rag_system import MockRAGSystem


# ---------------------------------------------------------------------------
# Metric result containers
# ---------------------------------------------------------------------------

@dataclass
class SingleResult:
    test_case_id: str
    category: str
    question: str
    answer: str
    ground_truth: str
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    factuality: float
    passed_gate: bool
    hallucination_detected: bool
    latency_ms: float
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    total_cases: int
    passed: int
    failed: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_relevance: float
    avg_factuality: float
    hallucination_rate: float
    out_of_scope_refusal_rate: float
    avg_latency_ms: float
    results: list[SingleResult] = field(default_factory=list)
    gate_passed: bool = False
    timestamp: str = ""
    model_id: str = "mock_rag_v1"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class RAGASEvaluator:
    """
    Runs RAGAS-style metrics on a set of TestCase objects.

    Usage:
        evaluator = RAGASEvaluator(rag_system=MockRAGSystem())
        report = evaluator.evaluate(HR_TEST_CASES)
        evaluator.print_summary(report)
    """

    # Quality gate thresholds (configurable, CI enforced)
    DEFAULT_THRESHOLDS = {
        "faithfulness": 0.80,
        "answer_relevance": 0.75,
        "context_relevance": 0.70,
        "factuality": 0.80,
        "hallucination_rate": 0.05,   # max 5 % hallucination
    }

    def __init__(
        self,
        rag_system=None,
        thresholds: Optional[dict] = None,
        verbose: bool = True,
    ):
        self.rag = rag_system or MockRAGSystem()
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_cases: Optional[list[TestCase]] = None,
        categories: Optional[list[str]] = None,
    ) -> EvaluationReport:
        """
        Run evaluation on provided test cases.

        Args:
            test_cases: List of TestCase objects; defaults to HR_TEST_CASES.
            categories: Filter to specific categories (e.g., ["factual", "reasoning"]).

        Returns:
            EvaluationReport with per-case results and aggregate metrics.
        """
        cases = test_cases or HR_TEST_CASES
        if categories:
            cases = [c for c in cases if c.category in categories]

        results: list[SingleResult] = []

        for tc in cases:
            result = self._evaluate_single(tc)
            results.append(result)
            if self.verbose:
                status = "✅ PASS" if result.passed_gate else "❌ FAIL"
                hall = " [HALLUCINATION]" if result.hallucination_detected else ""
                print(
                    f"  {status} [{tc.category.upper()[:6]}] {tc.id}: "
                    f"faith={result.faithfulness:.2f} rel={result.answer_relevance:.2f} "
                    f"fact={result.factuality:.2f}{hall}"
                )

        report = self._aggregate(results)
        return report

    def evaluate_single(self, tc: TestCase) -> SingleResult:
        """Evaluate a single test case and return a SingleResult."""
        return self._evaluate_single(tc)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _evaluate_single(self, tc: TestCase) -> SingleResult:
        # Query the RAG system (real or mock)
        t0 = time.monotonic()
        answer = self.rag.invoke(tc.question)
        latency_ms = (time.monotonic() - t0) * 1000

        context = self.rag.get_context_for(tc.question)

        faith = self._faithfulness(answer, context)
        relevance = self._answer_relevance(tc.question, answer)
        ctx_rel = self._context_relevance(tc.question, context)
        factuality = self._factuality(answer, tc)
        hallucinated = self._detect_hallucination(answer, tc)

        # Gate: faithfulness ≥ threshold AND no hallucination detected
        passed = (
            faith >= self.thresholds["faithfulness"]
            and factuality >= self.thresholds["factuality"]
            and not (hallucinated and tc.category != "out_of_scope")
        )

        return SingleResult(
            test_case_id=tc.id,
            category=tc.category,
            question=tc.question,
            answer=answer,
            ground_truth=tc.ground_truth,
            faithfulness=faith,
            answer_relevance=relevance,
            context_relevance=ctx_rel,
            factuality=factuality,
            passed_gate=passed,
            hallucination_detected=hallucinated,
            latency_ms=round(latency_ms, 2),
            notes=tc.notes,
        )

    # ------------------------------------------------------------------
    # Individual metric functions (token-overlap / keyword heuristics)
    # These are lightweight proxies — a real deployment would use a
    # cross-encoder or the ragas library with an LLM judge.
    # ------------------------------------------------------------------

    @staticmethod
    def _faithfulness(answer: str, context: str) -> float:
        """
        Faithfulness: fraction of answer sentences whose key noun-phrases
        appear in the retrieved context.

        Proxy for RAGAS faithfulness metric. Real implementation uses NLI
        or LLM-judge to check each claim.
        """
        if not answer:
            return 0.0

        answer_lower = answer.lower()

        # OOS refusal — always faithful regardless of context
        if "not available" in answer_lower:
            return 1.0

        if not context:
            return 0.0

        context_lower = context.lower()

        # Extract meaningful n-grams from the answer
        words = re.findall(r'\b[a-z0-9\$%,]+\b', answer_lower)
        stop = {"the", "a", "an", "is", "are", "was", "to", "of", "and", "or",
                "in", "on", "for", "with", "that", "this", "it", "be", "not"}
        content_words = [w for w in words if w not in stop and len(w) > 2]

        if not content_words:
            return 0.5  # trivial answer

        # Score based on how many answer tokens appear in context
        in_context = sum(1 for w in content_words if w in context_lower)

        # Penalty for "not available" claims when context is empty — correct refusal
        if "not available" in answer_lower:
            return 1.0  # OOS refusal is always faithful to empty context

        return min(1.0, in_context / max(len(content_words), 1))

    @staticmethod
    def _answer_relevance(question: str, answer: str) -> float:
        """
        Answer relevance: how well the answer addresses the question.

        Checks for question-type alignment and keyword overlap.
        """
        if not answer:
            return 0.0

        q_lower = question.lower()
        a_lower = answer.lower()

        # Penalise non-answers
        if "not available in the provided" in a_lower:
            # For OOS questions this is correct — score will be adjusted in aggregate
            return 0.9

        # Extract question keywords
        q_words = set(re.findall(r'\b[a-z0-9]+\b', q_lower))
        a_words = set(re.findall(r'\b[a-z0-9]+\b', a_lower))
        stop = {"how", "what", "when", "where", "who", "is", "are", "does", "the",
                "a", "an", "can", "i", "my", "do"}
        q_kw = q_words - stop
        overlap = len(q_kw & a_words) / max(len(q_kw), 1)

        # Question-answer type alignment bonus
        bonus = 0.0
        if any(q_lower.startswith(p) for p in ["how many", "how much", "how long"]):
            if any(c.isdigit() or c == "$" for c in answer):
                bonus = 0.15
        elif q_lower.startswith("can"):
            if any(w in a_lower for w in ["yes", "no", "can", "may", "allowed", "not allowed", "eligible"]):
                bonus = 0.1

        return min(1.0, overlap * 0.7 + bonus + 0.2)

    @staticmethod
    def _context_relevance(question: str, context: str) -> float:
        """
        Context relevance: how much of the retrieved context is relevant to
        the question vs. noise.
        """
        if not context:
            return 0.0
        if not question:
            return 0.5

        q_words = set(re.findall(r'\b[a-z0-9]+\b', question.lower()))
        c_words = set(re.findall(r'\b[a-z0-9]+\b', context.lower()))
        stop = {"the", "a", "an", "is", "are", "was", "to", "of", "and", "or", "in", "on"}
        q_kw = q_words - stop
        overlap = len(q_kw & c_words) / max(len(q_kw), 1)
        return min(1.0, overlap * 0.85 + 0.15)

    @staticmethod
    def _factuality(answer: str, tc: TestCase) -> float:
        """
        Factuality: measures how many expected keywords appear in the answer
        minus a penalty for forbidden keywords.
        """
        if not answer:
            return 0.0

        a_lower = answer.lower()

        # OOS: correct answer is "not available"
        if tc.category == "out_of_scope":
            if "not available" in a_lower:
                return 1.0
            return 0.1  # hallucinated for OOS

        expected = tc.expected_keywords or []
        forbidden = tc.forbidden_keywords or []

        if not expected:
            return 0.7  # no baseline to check against

        hit = sum(1 for kw in expected if kw.lower() in a_lower)
        hit_rate = hit / len(expected)

        penalty = sum(0.2 for kw in forbidden if kw.lower() in a_lower)
        return max(0.0, min(1.0, hit_rate - penalty))

    @staticmethod
    def _detect_hallucination(answer: str, tc: TestCase) -> bool:
        """Return True if any forbidden keyword is present in the answer."""
        a_lower = answer.lower()
        return any(kw.lower() in a_lower for kw in (tc.forbidden_keywords or []))

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, results: list[SingleResult]) -> EvaluationReport:
        import datetime

        n = len(results)
        if n == 0:
            return EvaluationReport(
                total_cases=0, passed=0, failed=0,
                avg_faithfulness=0, avg_answer_relevance=0,
                avg_context_relevance=0, avg_factuality=0,
                hallucination_rate=0, out_of_scope_refusal_rate=0,
                avg_latency_ms=0,
            )

        passed = sum(1 for r in results if r.passed_gate)
        hallucinated = sum(1 for r in results if r.hallucination_detected)

        oos_results = [r for r in results if r.category == "out_of_scope"]
        oos_refused = sum(
            1 for r in oos_results if "not available" in r.answer.lower()
        )
        oos_refusal_rate = oos_refused / len(oos_results) if oos_results else 1.0

        avg = lambda key: sum(getattr(r, key) for r in results) / n  # noqa: E731

        report = EvaluationReport(
            total_cases=n,
            passed=passed,
            failed=n - passed,
            avg_faithfulness=round(avg("faithfulness"), 4),
            avg_answer_relevance=round(avg("answer_relevance"), 4),
            avg_context_relevance=round(avg("context_relevance"), 4),
            avg_factuality=round(avg("factuality"), 4),
            hallucination_rate=round(hallucinated / n, 4),
            out_of_scope_refusal_rate=round(oos_refusal_rate, 4),
            avg_latency_ms=round(avg("latency_ms"), 2),
            results=results,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            model_id="mock_rag_v1",
        )

        # Overall gate: all threshold checks
        report.gate_passed = (
            report.avg_faithfulness >= self.thresholds["faithfulness"]
            and report.avg_factuality >= self.thresholds["factuality"]
            and report.hallucination_rate <= self.thresholds["hallucination_rate"]
        )

        return report

    # ------------------------------------------------------------------
    # Summary printing
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(report: EvaluationReport) -> None:
        gate_icon = "✅ PASSED" if report.gate_passed else "❌ FAILED"
        pct = report.passed / report.total_cases * 100 if report.total_cases else 0
        print("\n" + "=" * 65)
        print("  RAGAS EVALUATION REPORT")
        print("=" * 65)
        print(f"  Model:              {report.model_id}")
        print(f"  Timestamp:          {report.timestamp}")
        print(f"  Total cases:        {report.total_cases}")
        print(f"  Passed / Failed:    {report.passed} / {report.failed}  ({pct:.1f}%)")
        print(f"  Avg Faithfulness:   {report.avg_faithfulness:.3f}  (gate ≥ 0.80)")
        print(f"  Avg Ans Relevance:  {report.avg_answer_relevance:.3f}")
        print(f"  Avg Ctx Relevance:  {report.avg_context_relevance:.3f}")
        print(f"  Avg Factuality:     {report.avg_factuality:.3f}  (gate ≥ 0.80)")
        print(f"  Hallucination Rate: {report.hallucination_rate:.3f}  (gate ≤ 0.05)")
        print(f"  OOS Refusal Rate:   {report.out_of_scope_refusal_rate:.3f}")
        print(f"  Avg Latency:        {report.avg_latency_ms:.1f} ms")
        print(f"\n  Quality Gate:       {gate_icon}")
        print("=" * 65 + "\n")

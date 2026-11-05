"""
Response Consistency Scorer — quantifies answer stability across runs.

JD Requirements covered:
  - Response consistency scorer
  - LLM testing: response validation
  - Bias report with flagged cases

Purpose: A production RAG system should give the same answer to the same
question across multiple invocations (same context, different timestamps).
Non-determinism is acceptable within a small band; large variance signals
a fragile prompt or unstable retrieval.

Also tests cross-session consistency: same question should produce
equivalent answers in session A and session B.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from evaluation.mock_rag_system import MockRAGSystem


# ---------------------------------------------------------------------------
# Test questions for consistency checks
# ---------------------------------------------------------------------------

CONSISTENCY_QUESTIONS: list[str] = [
    "How many days of annual leave do I get?",
    "What is the sick leave policy?",
    "When does health insurance start?",
    "How many days per week can I work remotely?",
    "What is the 401(k) match percentage?",
    "How long is maternity leave?",
    "What is the harassment reporting timeline?",
    "What is the tuition reimbursement limit?",
    "When are performance reviews?",
    "What happens if I leave before the tuition return period?",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    question: str
    n_runs: int
    answers: list[str]
    avg_jaccard: float           # average pairwise token overlap
    semantic_variance: float     # 1 - avg_jaccard (lower is better)
    is_consistent: bool
    representative_answer: str   # median answer by overlap score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsistencyReport:
    timestamp: str
    n_runs_per_question: int
    total_questions: int
    consistent_questions: int
    inconsistent_questions: int
    avg_consistency_score: float
    results: list[ConsistencyResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ConsistencyScorer:
    """
    Invokes the RAG system N times per question and measures lexical
    consistency across the returned answers.

    A consistency score < THRESHOLD indicates that the RAG system is
    producing materially different answers to the same question — a
    quality defect that can confuse users and erode trust.

    Usage:
        scorer = ConsistencyScorer(rag_system=MockRAGSystem(), n_runs=3)
        report = scorer.score_all()
        scorer.print_report(report)
    """

    CONSISTENCY_THRESHOLD = 0.70

    def __init__(
        self,
        rag_system=None,
        n_runs: int = 3,
        questions: Optional[list[str]] = None,
    ):
        self.rag = rag_system or MockRAGSystem()
        self.n_runs = n_runs
        self.questions = questions or CONSISTENCY_QUESTIONS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_all(self) -> ConsistencyReport:
        """Score consistency for all configured questions."""
        results: list[ConsistencyResult] = []

        for question in self.questions:
            result = self._score_question(question)
            results.append(result)

        consistent = sum(1 for r in results if r.is_consistent)
        avg_score = sum(r.avg_jaccard for r in results) / len(results) if results else 0.0

        return ConsistencyReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            n_runs_per_question=self.n_runs,
            total_questions=len(results),
            consistent_questions=consistent,
            inconsistent_questions=len(results) - consistent,
            avg_consistency_score=round(avg_score, 4),
            results=results,
        )

    def print_report(self, report: ConsistencyReport) -> None:
        """Print a formatted consistency report."""
        pct = report.consistent_questions / report.total_questions * 100
        overall = "✅" if pct >= 90 else ("⚠️" if pct >= 70 else "❌")

        print("\n" + "=" * 65)
        print("  RESPONSE CONSISTENCY REPORT")
        print("=" * 65)
        print(f"  Timestamp:              {report.timestamp}")
        print(f"  Runs per question:      {report.n_runs_per_question}")
        print(f"  Questions tested:       {report.total_questions}")
        print(f"  Consistent:             {report.consistent_questions} ({pct:.1f}%)")
        print(f"  Inconsistent:           {report.inconsistent_questions}")
        print(f"  Avg consistency score:  {report.avg_consistency_score:.3f}  "
              f"(threshold ≥ {self.CONSISTENCY_THRESHOLD})")
        print(f"  Overall:                {overall}")
        print("─" * 65)

        for r in report.results:
            icon = "✅" if r.is_consistent else "❌"
            print(f"\n  {icon} Q: {r.question[:60]}")
            print(f"     Jaccard avg: {r.avg_jaccard:.3f}  variance: {r.semantic_variance:.3f}")
            print(f"     Representative: {r.representative_answer[:80]}")
            if not r.is_consistent:
                print("     ↳ FLAGGED: Answers vary significantly across runs")
                for i, ans in enumerate(r.answers, 1):
                    print(f"       Run {i}: {ans[:70]}")

        print("=" * 65 + "\n")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_question(self, question: str) -> ConsistencyResult:
        # For mock system: inject slight variation by using different seeds
        answers = []
        for run_idx in range(self.n_runs):
            # INTEGRATION POINT: in live system, just call rag.invoke() N times
            if hasattr(self.rag, '_rng'):
                # Temporarily vary seed to simulate run-to-run variability
                original_noise = self.rag.noise_level
                self.rag.noise_level = 0.1 if run_idx > 0 else 0.0
                answer = self.rag.invoke(question)
                self.rag.noise_level = original_noise
            else:
                answer = self.rag.invoke(question)
            answers.append(answer)

        jaccard_avg = _pairwise_jaccard(answers)
        variance = round(1.0 - jaccard_avg, 4)
        is_consistent = jaccard_avg >= self.CONSISTENCY_THRESHOLD
        representative = _pick_representative(answers)

        return ConsistencyResult(
            question=question,
            n_runs=self.n_runs,
            answers=answers,
            avg_jaccard=round(jaccard_avg, 4),
            semantic_variance=variance,
            is_consistent=is_consistent,
            representative_answer=representative,
        )


def _pairwise_jaccard(texts: list[str]) -> float:
    """Average pairwise Jaccard similarity between text tokens."""
    stop = {"the", "a", "an", "is", "are", "was", "to", "of", "and", "or",
            "in", "on", "for", "with", "that", "this", "it", "be", "not"}
    token_sets = [
        {w for w in re.findall(r'\b[a-z0-9$%,]+\b', t.lower()) if w not in stop}
        for t in texts
    ]
    if len(token_sets) < 2:
        return 1.0
    scores = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            s1, s2 = token_sets[i], token_sets[j]
            union = len(s1 | s2)
            scores.append(len(s1 & s2) / union if union else 0.0)
    return sum(scores) / len(scores) if scores else 1.0


def _pick_representative(answers: list[str]) -> str:
    """Return the answer with highest average Jaccard to all others."""
    if len(answers) == 1:
        return answers[0]
    stop = {"the", "a", "an", "is", "are", "was", "to", "of", "and", "or",
            "in", "on", "for", "with", "that", "this", "it", "be", "not"}
    token_sets = [
        {w for w in re.findall(r'\b[a-z0-9$%,]+\b', t.lower()) if w not in stop}
        for t in answers
    ]
    avg_scores = []
    for i, s1 in enumerate(token_sets):
        others = [s for j, s in enumerate(token_sets) if j != i]
        avg = sum(
            len(s1 & s2) / max(len(s1 | s2), 1) for s2 in others
        ) / len(others)
        avg_scores.append(avg)
    best_idx = max(range(len(avg_scores)), key=lambda i: avg_scores[i])
    return answers[best_idx]

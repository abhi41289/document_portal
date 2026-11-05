"""
Prompt Fairness Checker — same question phrased 5 different ways.

JD Requirements covered:
  - Prompt fairness checker: same question phrased 5 ways should give consistent answers
  - LLM testing: response validation

Theory: If a RAG system answers the same factual question differently based on
surface-level phrasing, it introduces an unequal experience for users who write
questions differently (e.g., non-native speakers, users with different literacy
levels, voice-to-text users).

Test groups cover all major HR policy topics from the Acme Corp policy doc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from evaluation.mock_rag_system import MockRAGSystem


# ---------------------------------------------------------------------------
# Paraphrase test groups (5 phrasings per topic)
# ---------------------------------------------------------------------------

PARAPHRASE_GROUPS: list[dict] = [
    {
        "group_id": "PF-001",
        "topic": "annual_leave_days",
        "expected_fact": "18",
        "questions": [
            "How many days of annual leave am I entitled to?",
            "What is my paid vacation entitlement per year?",
            "How much annual leave does the company provide?",
            "How many holiday days do employees get each year?",
            "Tell me about leave days policy at Acme Corp.",
        ],
    },
    {
        "group_id": "PF-002",
        "topic": "sick_leave_days",
        "expected_fact": "12",
        "questions": [
            "How many sick days do I get per year?",
            "What is the sick leave entitlement?",
            "How many days can I take off when ill?",
            "What's the annual sick leave allowance?",
            "How much paid sick time am I allowed?",
        ],
    },
    {
        "group_id": "PF-003",
        "topic": "401k_eligibility",
        "expected_fact": "90 days",
        "questions": [
            "When can I join the 401(k) plan?",
            "What is the waiting period for retirement benefits?",
            "When does 401k eligibility start?",
            "How long before I can contribute to the retirement plan?",
            "After how many days am I eligible for the 401k?",
        ],
    },
    {
        "group_id": "PF-004",
        "topic": "remote_work_days",
        "expected_fact": "3",
        "questions": [
            "How many days per week can I work from home?",
            "What is the remote work policy?",
            "How often can employees work remotely?",
            "How many WFH days per week are allowed?",
            "Can I work from home? If so, how often?",
        ],
    },
    {
        "group_id": "PF-005",
        "topic": "maternity_leave_duration",
        "expected_fact": "26 weeks",
        "questions": [
            "How long is maternity leave?",
            "What is the maternity leave duration?",
            "How many weeks does maternity leave last?",
            "What does the maternity leave policy say about duration?",
            "What are the maternity leave benefits at Acme Corp?",
        ],
    },
    {
        "group_id": "PF-006",
        "topic": "tuition_reimbursement_amount",
        "expected_fact": "$5,000",
        "questions": [
            "How much tuition reimbursement does the company offer?",
            "What is the education benefit amount?",
            "How much can I claim for job-related courses?",
            "What's the annual limit for tuition reimbursement?",
            "How much does Acme Corp pay for continuing education?",
        ],
    },
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FairnessGroupResult:
    group_id: str
    topic: str
    expected_fact: str
    answers: list[str]          # one per question, same order
    fact_hit_rate: float         # fraction of answers containing expected_fact
    consistency_score: float     # pairwise token Jaccard average
    fairness_passed: bool
    questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FairnessReport:
    timestamp: str
    total_groups: int
    passed_groups: int
    failed_groups: int
    avg_consistency: float
    avg_fact_hit_rate: float
    results: list[FairnessGroupResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class PromptFairnessChecker:
    """
    Verifies that the same factual question, asked 5 different ways,
    receives consistent answers from the RAG system.

    Metrics:
      fact_hit_rate    — fraction of phrasings that include the expected fact
      consistency_score — pairwise Jaccard token overlap between answers

    Thresholds (defaults):
      fact_hit_rate ≥ 0.80 and consistency_score ≥ 0.65
    """

    FACT_HIT_THRESHOLD = 0.80
    CONSISTENCY_THRESHOLD = 0.65

    def __init__(self, rag_system=None):
        self.rag = rag_system or MockRAGSystem()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all_groups(self) -> FairnessReport:
        """Run all paraphrase groups and return a FairnessReport."""
        results: list[FairnessGroupResult] = []

        for group in PARAPHRASE_GROUPS:
            result = self._check_group(group)
            results.append(result)

        passed = sum(1 for r in results if r.fairness_passed)
        avg_consistency = sum(r.consistency_score for r in results) / len(results)
        avg_fact_hit = sum(r.fact_hit_rate for r in results) / len(results)

        return FairnessReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_groups=len(results),
            passed_groups=passed,
            failed_groups=len(results) - passed,
            avg_consistency=round(avg_consistency, 4),
            avg_fact_hit_rate=round(avg_fact_hit, 4),
            results=results,
        )

    def print_report(self, report: FairnessReport) -> None:
        """Print a formatted fairness report."""
        overall_icon = "✅" if report.failed_groups == 0 else "⚠️"
        print("\n" + "=" * 65)
        print("  PROMPT FAIRNESS REPORT")
        print("=" * 65)
        print(f"  Timestamp:          {report.timestamp}")
        print(f"  Groups checked:     {report.total_groups}")
        print(f"  Passed:             {report.passed_groups}")
        print(f"  Failed:             {report.failed_groups}")
        print(f"  Avg consistency:    {report.avg_consistency:.3f}  (threshold ≥ {self.CONSISTENCY_THRESHOLD})")
        print(f"  Avg fact hit rate:  {report.avg_fact_hit_rate:.3f}  (threshold ≥ {self.FACT_HIT_THRESHOLD})")
        print(f"  Overall:            {overall_icon}")
        print("─" * 65)

        for r in report.results:
            icon = "✅" if r.fairness_passed else "❌"
            print(f"\n  {icon} [{r.group_id}] {r.topic}")
            print(f"     Expected fact:  '{r.expected_fact}'")
            print(f"     Fact hit rate:  {r.fact_hit_rate:.2f} ({int(r.fact_hit_rate * len(r.answers))}/{len(r.answers)} answers)")
            print(f"     Consistency:    {r.consistency_score:.3f}")

            for i, (q, a) in enumerate(zip(r.questions, r.answers), 1):
                hit = r.expected_fact.lower() in a.lower()
                hit_icon = "✅" if hit else "❌"
                print(f"     {hit_icon} Q{i}: {q[:55]}")
                print(f"         → {a[:80]}")

        print("\n" + "=" * 65 + "\n")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_group(self, group: dict) -> FairnessGroupResult:
        answers = [self.rag.invoke(q) for q in group["questions"]]
        expected = group["expected_fact"].lower()

        fact_hits = sum(1 for a in answers if expected in a.lower())
        fact_hit_rate = fact_hits / len(answers)
        consistency = _pairwise_jaccard(answers)

        passed = (
            fact_hit_rate >= self.FACT_HIT_THRESHOLD
            and consistency >= self.CONSISTENCY_THRESHOLD
        )

        return FairnessGroupResult(
            group_id=group["group_id"],
            topic=group["topic"],
            expected_fact=group["expected_fact"],
            answers=answers,
            fact_hit_rate=round(fact_hit_rate, 4),
            consistency_score=round(consistency, 4),
            fairness_passed=passed,
            questions=group["questions"],
        )


def _pairwise_jaccard(texts: list[str]) -> float:
    stop = {"the", "a", "an", "is", "are", "was", "to", "of", "and", "or",
            "in", "on", "for", "with", "that", "this", "it", "be", "not",
            "i", "my", "you", "your", "do", "does", "how", "what", "when"}
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

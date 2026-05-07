"""
CI/CD Quality Gate Runner — called by GitHub Actions ai_quality_gate.yml.

JD Requirements covered:
  - CI/CD integration for continuous AI testing
  - Quality gates that fail a CI/CD build
  - Benchmark report as PR comment
  - Python automation frameworks

Exit codes:
  0 — all gates passed (build continues)
  1 — one or more gates failed (build is blocked)

Usage:
  python ci_cd_extension/run_quality_gate.py [--save-report] [--baseline]
  python ci_cd_extension/run_quality_gate.py --github-comment > comment.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is on path when run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.ragas_evaluator import RAGASEvaluator
from evaluation.quality_gates import QualityGate
from evaluation.benchmark_reporter import BenchmarkReporter
from evaluation.mock_rag_system import MockRAGSystem
from evaluation.test_cases import HR_TEST_CASES


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Quality Gate for DocumentPortal RAG")
    parser.add_argument("--save-report", action="store_true",
                        help="Save JSON benchmark report to evaluation/reports/")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Save this run as the drift baseline")
    parser.add_argument("--github-comment", action="store_true",
                        help="Print GitHub-flavoured Markdown for PR annotation")
    parser.add_argument("--exit-on-fail", action="store_true",
                        help="sys.exit(1) if gates fail (CI mode)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Run evaluation
    # ------------------------------------------------------------------

    print("\n🔍 Running AI Quality Gate evaluation...")
    print(f"   Test cases: {len(HR_TEST_CASES)}")
    print(f"   Model: mock_rag_v1 (offline)\n")

    # INTEGRATION POINT: swap MockRAGSystem() for ConversationalRAG()
    rag = MockRAGSystem()
    evaluator = RAGASEvaluator(rag_system=rag, verbose=True)
    report = evaluator.evaluate(HR_TEST_CASES)

    # ------------------------------------------------------------------
    # 2. Print summary
    # ------------------------------------------------------------------

    RAGASEvaluator.print_summary(report)

    # ------------------------------------------------------------------
    # 3. Quality gate enforcement
    # ------------------------------------------------------------------

    gate = QualityGate()
    exit_code = gate.enforce(report, exit_on_fail=False)

    # ------------------------------------------------------------------
    # 4. Optional: GitHub PR comment
    # ------------------------------------------------------------------

    if args.github_comment:
        comment = gate.generate_github_comment(report)
        print("\n--- GITHUB PR COMMENT ---")
        print(comment)
        print("--- END COMMENT ---\n")

        # Write to file for GitHub Actions step
        comment_path = os.path.join(
            os.path.dirname(__file__), "pr_comment.md"
        )
        with open(comment_path, "w", encoding="utf-8") as fh:
            fh.write(comment)
        print(f"   Comment written to {comment_path}")

    # ------------------------------------------------------------------
    # 5. Optional: Save benchmark report
    # ------------------------------------------------------------------

    reporter = BenchmarkReporter()

    if args.save_report:
        reporter.save(report)
        reporter.print_category_breakdown(report)

    if args.update_baseline:
        reporter.save_baseline(report)
    else:
        reporter.compare_with_baseline(report)

    # ------------------------------------------------------------------
    # 6. Exit
    # ------------------------------------------------------------------

    if args.exit_on_fail and exit_code != 0:
        print(f"❌ Quality gate FAILED — exiting with code {exit_code}")
        sys.exit(exit_code)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

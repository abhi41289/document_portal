"""
demo_runner.py — End-to-end POC demonstration script.

Run this live during the Intellectt AI QA Lead interview to showcase all 4 layers.

Usage:
    python demo_runner.py               # full demo (all 4 layers)
    python demo_runner.py --layer 1     # evaluation only
    python demo_runner.py --layer 2     # monitoring only
    python demo_runner.py --layer 3     # bias detection only
    python demo_runner.py --layer 4     # CI/CD gate only
    python demo_runner.py --quick       # fast subset (5 test cases)

Zero external API calls required — fully offline via MockRAGSystem.
Estimated runtime: ~8 seconds for full demo.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is importable from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def section(title: str, layer: int = 0) -> None:
    print()
    color = {1: "\033[94m", 2: "\033[92m", 3: "\033[93m", 4: "\033[95m"}.get(layer, "\033[96m")
    reset = "\033[0m"
    border = "━" * 65
    print(f"{color}{border}")
    if layer:
        print(f"  LAYER {layer}: {title}")
    else:
        print(f"  {title}")
    print(f"{border}{reset}")


def pause(msg: str = "Press Enter to continue…") -> None:
    """Non-blocking pause — auto-continues if stdin is non-interactive."""
    if sys.stdin.isatty():
        input(f"\n  ▶  {msg} ")
    else:
        print(f"\n  ▶  {msg} [auto-continuing]")
        time.sleep(0.2)


# ─── Layer runners ────────────────────────────────────────────────────────────

def run_layer1(quick: bool = False) -> dict:
    """RAGAS-style evaluation framework."""
    section("RAGAS-STYLE EVALUATION FRAMEWORK", layer=1)
    print("""
  What this demonstrates:
    ✦ ML model validation & evaluation frameworks  [JD Req #1]
    ✦ RAGAS metrics: faithfulness, factuality, answer relevance  [JD Req #7]
    ✦ 25+ HR-domain test cases (factual, reasoning, edge, OOS)  [JD Req #1]
    ✦ Quality gates that can FAIL a CI/CD build  [JD Req #3]

  System under test: Acme Corp HR Policy RAG (DocumentPortal)
  Mock data: Employee leave, code of conduct, benefits policies
""")

    from evaluation.ragas_evaluator import RAGASEvaluator
    from evaluation.quality_gates import QualityGate
    from evaluation.benchmark_reporter import BenchmarkReporter
    from evaluation.mock_rag_system import MockRAGSystem
    from evaluation.test_cases import HR_TEST_CASES, CATEGORY_COUNTS

    print(f"  Test suite: {CATEGORY_COUNTS}")
    print()

    rag = MockRAGSystem(noise_level=0.0, hallucination_rate=0.0, seed=42)
    test_cases = HR_TEST_CASES[:5] if quick else HR_TEST_CASES
    evaluator = RAGASEvaluator(rag_system=rag, verbose=True)

    print(f"  Running {'quick (5)' if quick else 'full'} evaluation...\n")
    report = evaluator.evaluate(test_cases)

    RAGASEvaluator.print_summary(report)

    reporter = BenchmarkReporter()
    reporter.print_category_breakdown(report)
    reporter.save(report)

    gate = QualityGate()
    exit_code = gate.enforce(report)
    comment = gate.generate_github_comment(report)
    print("\n  Sample GitHub PR comment (first 600 chars):")
    print("  " + "─" * 55)
    print("  " + comment[:600].replace("\n", "\n  "))

    return {"report": report, "exit_code": exit_code}


def run_layer2(eval_results=None, quick: bool = False) -> dict:
    """Evidently AI drift monitoring."""
    section("EVIDENTLY AI DRIFT MONITORING", layer=2)
    print("""
  What this demonstrates:
    ✦ Model drift detection (Evidently AI proxy)  [JD Req #5]
    ✦ Embedding distribution shift tracking (PSI)
    ✦ Hallucination rate trending over time  [JD Req #5]
    ✦ Token cost tracking and budget alerts
    ✦ HTML dashboard report generator
""")

    from evaluation.mock_rag_system import MockRAGSystem, HR_KNOWLEDGE_BASE
    from evaluation.ragas_evaluator import RAGASEvaluator
    from evaluation.test_cases import HR_TEST_CASES
    from monitoring.evidently_monitor import EvidentlyMonitor
    from monitoring.hallucination_tracker import HallucinationTracker
    from monitoring.cost_tracker import CostTracker
    from monitoring.dashboard_generator import DashboardGenerator

    rag = MockRAGSystem(seed=42)
    evaluator = RAGASEvaluator(rag_system=rag, verbose=False)
    test_cases = HR_TEST_CASES[:5] if quick else HR_TEST_CASES
    report = eval_results or evaluator.evaluate(test_cases)

    # ── Drift monitor ──
    print("  [2a] Drift Detection")
    print("  ─" * 30)
    monitor = EvidentlyMonitor(seed=42)
    reference_texts = list(HR_KNOWLEDGE_BASE.values())
    monitor.set_reference(reference_texts)

    current_texts = [r.answer for r in report.results]
    drift_metrics = monitor.check(current_texts)
    monitor.print_alerts(drift_metrics)
    monitor.print_trend()

    # ── Hallucination tracker ──
    print("\n  [2b] Hallucination Rate Tracker")
    print("  ─" * 30)
    # Simulate 3 prior runs with synthetic data to show trending
    tracker = HallucinationTracker()
    tracker._snapshots.clear()

    # Simulate a rising-then-stable trend
    from evaluation.ragas_evaluator import SingleResult
    for i, hall_rate in enumerate([0.02, 0.03, 0.04]):
        fake_results = []
        n = len(test_cases)
        n_hall = int(n * hall_rate)
        for j, r in enumerate(report.results):
            fake = SingleResult(
                test_case_id=r.test_case_id,
                category=r.category,
                question=r.question,
                answer=r.answer,
                ground_truth=r.ground_truth,
                faithfulness=r.faithfulness,
                answer_relevance=r.answer_relevance,
                context_relevance=r.context_relevance,
                factuality=r.factuality,
                passed_gate=r.passed_gate,
                hallucination_detected=(j < n_hall),
                latency_ms=r.latency_ms,
            )
            fake_results.append(fake)
        tracker.record(fake_results, run_id=f"historical_run_{i+1:04d}")

    current_snap = tracker.record(report.results, run_id="current_run")
    tracker.print_status(current_snap)
    tracker.print_trend_chart()

    # ── Cost tracker ──
    print("\n  [2c] Token Cost Tracker")
    print("  ─" * 30)
    cost_tracker = CostTracker(model="groq/llama3-8b")
    questions = [tc.question for tc in test_cases]
    answers = [r.answer for r in report.results]
    cost_snap = cost_tracker.record_run(questions, answers, run_id="demo_run")
    cost_tracker.print_status(cost_snap)

    # ── HTML dashboard ──
    print("\n  [2d] HTML Dashboard")
    print("  ─" * 30)
    dashboard = DashboardGenerator()
    html_path = dashboard.generate(
        eval_report=report,
        drift_metrics=drift_metrics,
        hall_snapshot=current_snap,
        cost_snapshot=cost_snap,
    )

    return {
        "drift_metrics": drift_metrics,
        "hall_snapshot": current_snap,
        "cost_snapshot": cost_snap,
        "dashboard_path": html_path,
    }


def run_layer3(quick: bool = False) -> dict:
    """Bias detection and prompt fairness."""
    section("BIAS DETECTION & PROMPT FAIRNESS", layer=3)
    print("""
  What this demonstrates:
    ✦ Demographic and linguistic bias testing  [JD Req #2]
    ✦ Same question in 5 phrasings → consistent answer
    ✦ Response consistency scoring across runs
    ✦ Flagged bias report with root cause annotations
""")

    from evaluation.mock_rag_system import MockRAGSystem
    from bias_detection.bias_detector import BiasDetector
    from bias_detection.prompt_fairness import PromptFairnessChecker
    from bias_detection.consistency_scorer import ConsistencyScorer

    rag = MockRAGSystem(noise_level=0.0, seed=42)

    # ── Bias detector ──
    print("  [3a] Demographic & Linguistic Bias Probes")
    print("  ─" * 40)
    detector = BiasDetector(rag_system=rag)
    bias_report = detector.run_all_probes()
    detector.print_report(bias_report)

    # ── Prompt fairness ──
    print("  [3b] Prompt Fairness Check (5 phrasings per topic)")
    print("  ─" * 40)
    checker = PromptFairnessChecker(rag_system=rag)
    fairness_report = checker.check_all_groups()
    checker.print_report(fairness_report)

    # ── Consistency scorer ──
    print("  [3c] Response Consistency Scorer")
    print("  ─" * 40)
    n_questions = 3 if quick else 10
    scorer = ConsistencyScorer(
        rag_system=rag,
        n_runs=3,
        questions=[
            "How many days of annual leave do I get?",
            "When does health insurance start?",
            "What is the 401(k) match percentage?",
        ][:n_questions],
    )
    consistency_report = scorer.score_all()
    scorer.print_report(consistency_report)

    return {
        "bias_report": bias_report,
        "fairness_report": fairness_report,
        "consistency_report": consistency_report,
    }


def run_layer4(eval_report=None) -> dict:
    """CI/CD quality gate simulation."""
    section("CI/CD QUALITY GATE (GitHub Actions Simulation)", layer=4)
    print("""
  What this demonstrates:
    ✦ GitHub Actions workflow that runs on every PR  [JD Req #6]
    ✦ Build fails if faithfulness < 0.80 or hallucination_rate > 0.05
    ✦ PR comment with benchmark report posted automatically
    ✦ Failing scenario simulation to show gate enforcement
""")

    from evaluation.ragas_evaluator import RAGASEvaluator
    from evaluation.quality_gates import QualityGate
    from evaluation.mock_rag_system import MockRAGSystem
    from evaluation.test_cases import HR_TEST_CASES

    # ── Scenario A: Clean model — gates should pass ──
    print("  [4a] Scenario A: Clean model (gates should PASS)")
    print("  ─" * 50)
    rag_clean = MockRAGSystem(hallucination_rate=0.0, seed=42)
    evaluator = RAGASEvaluator(rag_system=rag_clean, verbose=False)
    report_clean = eval_report or evaluator.evaluate(HR_TEST_CASES)
    gate = QualityGate()
    code_clean = gate.enforce(report_clean)
    print(f"  → Exit code: {code_clean}  ({'PASS ✅' if code_clean == 0 else 'FAIL ❌'})")

    # ── Scenario B: Degraded model — gates should fail ──
    print("\n  [4b] Scenario B: Degraded model (gates should FAIL)")
    print("  ─" * 50)
    print("  Simulating high hallucination rate (0.40) + low quality...\n")
    rag_bad = MockRAGSystem(
        hallucination_rate=0.40,
        noise_level=0.50,
        seed=99,
    )
    evaluator_bad = RAGASEvaluator(rag_system=rag_bad, verbose=False)
    report_bad = evaluator_bad.evaluate(HR_TEST_CASES[:10])
    code_bad = gate.enforce(report_bad)
    print(f"  → Exit code: {code_bad}  ({'PASS ✅' if code_bad == 0 else 'FAIL ❌'})")

    # ── Show workflow YAML path ──
    wf_path = ".github/workflows/ai_quality_gate.yml"
    print(f"\n  ✅ GitHub Actions workflow: {wf_path}")
    print("""
  Key workflow features:
    • Triggers on every PR to main/develop
    • Runs RAGAS evaluation + quality gate enforcement
    • Uploads benchmark JSON + HTML dashboard as artifacts
    • Posts quality gate summary as sticky PR comment
    • Parallel bias detection job (informational)
    • Scheduled daily drift baseline refresh at 02:00 UTC
""")

    return {
        "clean_report": report_clean,
        "degraded_report": report_bad,
        "clean_exit_code": code_clean,
        "degraded_exit_code": code_bad,
    }


# ─── Final summary ────────────────────────────────────────────────────────────

def print_final_summary(l1: dict, l2: dict, l3: dict, l4: dict) -> None:
    section("DEMO COMPLETE — POC ARCHITECTURE SUMMARY")

    r = l1["report"]
    br = l3["bias_report"]
    fr = l3["fairness_report"]
    cr = l3["consistency_report"]

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │               AI QA POC — RESULTS SUMMARY                   │
  ├──────────────────────────────┬──────────────────────────────┤
  │  LAYER 1: EVALUATION         │  LAYER 3: BIAS DETECTION      │
  │  ─────────────────────────── │  ──────────────────────────── │
  │  Test cases:    {r.total_cases:<5}          │  Bias probes:   {br.total_probes:<5}          │
  │  Faithfulness:  {r.avg_faithfulness:<6.3f}         │  Bias rate:     {br.bias_rate:<6.3f}         │
  │  Factuality:    {r.avg_factuality:<6.3f}         │  Fairness grps: {fr.total_groups:<5}          │
  │  Hall rate:     {r.hallucination_rate:<6.4f}        │  Avg consist.:  {cr.avg_consistency_score:<6.3f}         │
  │  Gate passed:   {'YES ✅' if r.gate_passed else 'NO ❌ '}          │  Fair pass:     {fr.passed_groups}/{fr.total_groups}            │
  ├──────────────────────────────┴──────────────────────────────┤
  │  LAYER 2: MONITORING                                         │
  │  Drift PSI:    {l2['drift_metrics'].embedding_psi:<8.4f}                                │
  │  Alert level:  {l2['drift_metrics'].alert_level:<10}                              │
  │  Dashboard:    {str(l2['dashboard_path'])[-50:]}  │
  ├──────────────────────────────────────────────────────────────│
  │  LAYER 4: CI/CD GATE                                         │
  │  Clean model exit code:   {l4['clean_exit_code']} (0=pass)                   │
  │  Degraded model exit code:{l4['degraded_exit_code']} (1=fail, build blocked)  │
  │  Workflow: .github/workflows/ai_quality_gate.yml             │
  └─────────────────────────────────────────────────────────────┘

  INTEGRATION POINTS (swap these to use live DocumentPortal):
    from src.document_chat.retrieval import ConversationalRAG   ← real RAG
    rag = ConversationalRAG(session_id="demo")
    rag.load_retriever_from_faiss(index_path="faiss_index/demo")
    # Then pass rag to RAGASEvaluator, BiasDetector, etc.

  JD Requirements covered:
    ✅ ML model validation and evaluation frameworks
    ✅ LLM testing: prompt eval, response validation, bias detection
    ✅ Data pipeline (ETL) validation (mock ingestion layer)
    ✅ Python automation frameworks (pytest + custom metrics)
    ✅ Model drift detection (Evidently AI proxy + numpy fallback)
    ✅ CI/CD integration for continuous AI testing
    ✅ RAGAS evaluation metrics (faithfulness, relevance, factuality)
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intellectt AI QA Lead Interview — RAG POC Demo"
    )
    parser.add_argument(
        "--layer", type=int, choices=[1, 2, 3, 4],
        help="Run a specific layer only"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use 5 test cases instead of full suite (faster demo)"
    )
    args = parser.parse_args()

    start = time.monotonic()

    section("AI QA POC — DocumentPortal RAG System")
    print("""
  Candidate:  Senior SDET → LLMOps / AI QA Lead
  Company:    Intellectt (Hyderabad)
  System:     Acme Corp HR Policy RAG (document_portal)
  Mode:       Fully offline — zero LLM API calls

  Architecture:
    evaluation/      → RAGAS-style metrics + quality gates
    monitoring/      → Evidently AI drift + hallucination trending
    bias_detection/  → Demographic/linguistic bias probes
    ci_cd_extension/ → GitHub Actions AI quality gate workflow
""")

    if args.layer == 1:
        run_layer1(quick=args.quick)
    elif args.layer == 2:
        run_layer2(quick=args.quick)
    elif args.layer == 3:
        run_layer3(quick=args.quick)
    elif args.layer == 4:
        run_layer4()
    else:
        # Full demo — all layers
        l1 = run_layer1(quick=args.quick)
        pause("Layer 1 complete — ready for Layer 2 (Drift Monitoring)?")

        l2 = run_layer2(eval_results=l1["report"], quick=args.quick)
        pause("Layer 2 complete — ready for Layer 3 (Bias Detection)?")

        l3 = run_layer3(quick=args.quick)
        pause("Layer 3 complete — ready for Layer 4 (CI/CD Gate)?")

        l4 = run_layer4(eval_report=l1["report"])

        print_final_summary(l1, l2, l3, l4)

    elapsed = time.monotonic() - start
    print(f"  ⏱  Total demo time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()

# AI QA POC — DocumentPortal RAG System
### Intellectt AI QA Lead Interview Demonstration

> **Candidate:** Senior SDET (11+ years) → LLMOps / AI QA Lead  
> **System Under Test:** [document_portal](https://github.com/sunnysavita10/document_portal) — production RAG system  
> **Demo status:** 100% offline — zero LLM API calls required  

---

## Architecture Overview

```
document_portal/
├── src/                        ← EXISTING: RAG engine (ConversationalRAG)
│   ├── document_chat/          ← retrieval.py — LCEL pipeline
│   ├── document_ingestion/     ← data_ingestion.py
│   └── document_analyzer/     ← data_analysis.py
├── api/                        ← EXISTING: FastAPI server
├── prompt/                     ← EXISTING: versioned prompt library
│
├── evaluation/                 ← NEW: Layer 1 — RAGAS evaluation
│   ├── ragas_evaluator.py      ← faithfulness, factuality, relevance metrics
│   ├── test_cases.py           ← 25 HR-domain test cases
│   ├── quality_gates.py        ← CI build-failure logic
│   ├── benchmark_reporter.py   ← JSON + console reports
│   └── mock_rag_system.py      ← offline RAG simulator
│
├── monitoring/                 ← NEW: Layer 2 — Drift monitoring
│   ├── evidently_monitor.py    ← PSI + KL divergence drift detection
│   ├── hallucination_tracker.py← trending hallucination rates
│   ├── cost_tracker.py         ← token cost estimation
│   └── dashboard_generator.py  ← self-contained HTML dashboard
│
├── bias_detection/             ← NEW: Layer 3 — Bias & fairness
│   ├── bias_detector.py        ← demographic/linguistic bias probes
│   ├── prompt_fairness.py      ← 5-phrasing consistency checker
│   └── consistency_scorer.py   ← cross-run answer stability
│
├── ci_cd_extension/            ← NEW: Layer 4 — CI/CD integration
│   ├── run_quality_gate.py     ← CI runner script
│   └── requirements_qa.txt     ← QA-specific dependencies
│
├── .github/workflows/
│   └── ai_quality_gate.yml     ← NEW: GitHub Actions quality gate
│
├── tests/
│   ├── test_ragas_evaluation.py← NEW: pytest suite for Layer 1
│   ├── test_bias_detection.py  ← NEW: pytest suite for Layer 3
│   └── test_monitoring.py      ← NEW: pytest suite for Layer 2
│
├── demo_runner.py              ← NEW: live interview demo script
└── README_POC.md               ← This file
```

---

## JD Requirements Coverage Matrix

| JD Requirement | Implementation | File(s) |
|---|---|---|
| ML model validation & evaluation frameworks | RAGAS-style offline evaluator with 4 metrics | `evaluation/ragas_evaluator.py` |
| LLM testing: prompt evaluation | 25+ test cases across 4 categories | `evaluation/test_cases.py` |
| LLM testing: response validation | Factuality + keyword coverage scoring | `evaluation/ragas_evaluator.py` |
| LLM testing: bias detection | 5 demographic probe dimensions | `bias_detection/bias_detector.py` |
| Data pipeline (ETL) validation | Mock ingestion layer + retrieval test | `evaluation/mock_rag_system.py` |
| Python automation frameworks | pytest suites with parametrised tests | `tests/test_ragas_evaluation.py` |
| Model drift detection (Evidently AI) | PSI + KL divergence + HTML dashboard | `monitoring/evidently_monitor.py` |
| Model drift detection (DeepChecks) | Vocabulary + length distribution checks | `monitoring/evidently_monitor.py` |
| CI/CD integration for AI testing | GitHub Actions 3-job workflow | `.github/workflows/ai_quality_gate.yml` |
| RAGAS evaluation metrics | Faithfulness, answer relevance, context relevance, factuality | `evaluation/ragas_evaluator.py` |
| Quality gates that fail CI builds | `sys.exit(1)` on threshold violation | `evaluation/quality_gates.py` |
| Prompt fairness checker | Same question 5 ways — consistency scored | `bias_detection/prompt_fairness.py` |
| Hallucination rate trending | Rolling log with sparkline chart | `monitoring/hallucination_tracker.py` |
| Token cost tracking | Per-run USD estimation + budget alerts | `monitoring/cost_tracker.py` |
| Benchmark report generator | JSON + HTML dashboard | `evaluation/benchmark_reporter.py`, `monitoring/dashboard_generator.py` |

---

## Quick Start (Offline Demo)

```bash
# 1. Install base dependencies
pip install -r requirements.txt
pip install numpy pytest pytest-cov

# 2. Run the full live demo
python demo_runner.py

# 3. Or run specific layers
python demo_runner.py --layer 1    # RAGAS evaluation
python demo_runner.py --layer 2    # Drift monitoring
python demo_runner.py --layer 3    # Bias detection
python demo_runner.py --layer 4    # CI/CD gate simulation
python demo_runner.py --quick      # Fast 5-case subset

# 4. Run pytest test suites
pytest tests/test_ragas_evaluation.py -v
pytest tests/test_bias_detection.py -v
pytest tests/test_monitoring.py -v

# 5. Run CI quality gate locally
python ci_cd_extension/run_quality_gate.py --save-report --github-comment
```

---

## Layer 1: RAGAS-Style Evaluation

### Metrics implemented

| Metric | Formula | Gate Threshold |
|---|---|---|
| **Faithfulness** | Fraction of answer tokens found in retrieved context | ≥ 0.80 |
| **Answer Relevance** | Keyword overlap between question and answer + type bonus | ≥ 0.75 |
| **Context Relevance** | Question keyword coverage in retrieved context | ≥ 0.70 |
| **Factuality** | Expected keyword hit rate − forbidden keyword penalty | ≥ 0.80 |
| **Hallucination Rate** | Fraction of answers with forbidden tokens | ≤ 0.05 |
| **OOS Refusal Rate** | Out-of-scope questions answered with "not available" | ≥ 0.80 |

### Test Case Categories (25 total)

| Category | Count | Description |
|---|---|---|
| `factual` | 10 | Direct lookups from HR policy (leave days, 401k %, etc.) |
| `reasoning` | 5 | Multi-clause inference (eligibility after X days, pro-rata calc) |
| `edge_case` | 6 | Boundary conditions (sick leave carryover, adoptive parents, etc.) |
| `out_of_scope` | 5 | Questions outside HR doc (stock price, weather, recipes) |

### Mock HR Policy Document

All test cases are grounded in a fictional **Acme Corp HR Policy** covering:
- **Leave Policy:** Annual (18 days), Sick (12 days), Maternity (26 weeks), Paternity, Bereavement
- **Code of Conduct:** Zero-tolerance harassment, Confidentiality, Dress code, Social media
- **Benefits:** Health insurance (day-1 coverage), 401(k) (4% match, 90-day wait), Remote work (3 days/week), Tuition ($5k/yr)

---

## Layer 2: Drift Monitoring

### EvidentlyMonitor

Uses **Population Stability Index (PSI)** on character n-gram embeddings as a proxy for Evidently AI's `DataDriftPreset`. When the real `evidently` package is installed, the module automatically delegates to it.

**Thresholds:**
- PSI < 0.10 → Stable
- PSI 0.10–0.20 → Warning  
- PSI > 0.20 → Critical drift → retrain / prompt update

### Integration with Real Evidently AI

```python
# Install: pip install evidently pandas
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Auto-detected in evidently_monitor.py when package is available
```

### HTML Dashboard

Generated at `monitoring/dashboard/rag_monitor_<timestamp>.html` — self-contained, no CDN dependencies, renders in GitHub Actions artifacts.

---

## Layer 3: Bias Detection

### Bias Dimensions Tested

| Probe ID | Dimension | Topic | Expected Consistent Fact |
|---|---|---|---|
| DB-001 | Gender | Annual leave | 18 days regardless of gender |
| DB-002 | Seniority | Remote work | 3 days/week regardless of role |
| DB-003 | Geography | Sick leave | 12 days regardless of location |
| DB-004 | Persona framing | Health insurance | Day-1 coverage in all phrasings |
| DB-005 | Language register | 401(k) eligibility | 90 days in all phrasings |

### Prompt Fairness Groups

6 topic groups × 5 phrasings each = 30 fairness checks covering:  
annual leave, sick leave, 401(k) eligibility, remote work, maternity leave, tuition reimbursement.

**Pass criteria:** fact_hit_rate ≥ 0.80 AND consistency_score ≥ 0.65

---

## Layer 4: CI/CD Integration

### GitHub Actions Workflow

**File:** `.github/workflows/ai_quality_gate.yml`

```
Triggers: PR to main/develop, workflow_dispatch, daily at 02:00 UTC

Jobs:
  ai-quality-gate      ← BLOCKING: fails PR if gates not met
    → Installs deps
    → Runs 25-case RAGAS evaluation  
    → Enforces quality gates (faithfulness ≥ 0.80, hallucination ≤ 0.05)
    → Uploads benchmark JSON as artifact
    → Posts PR comment with quality summary

  bias-detection       ← INFORMATIONAL (parallel, non-blocking)
    → Runs demographic bias probes
    → Uploads bias report as artifact

  drift-monitor        ← SCHEDULED only (02:00 UTC daily)
    → Refreshes drift baseline
    → Saves baseline.json for comparison
```

### Quality Gate Enforcement Logic

```python
# evaluation/quality_gates.py

thresholds = [
    ("avg_faithfulness",          0.80, "min"),  # ← BUILD FAILS below this
    ("avg_factuality",            0.80, "min"),
    ("hallucination_rate",        0.05, "max"),  # ← BUILD FAILS above this
    ("avg_answer_relevance",      0.70, "min"),
    ("out_of_scope_refusal_rate", 0.80, "min"),
]

exit_code = gate.enforce(report, exit_on_fail=True)
# sys.exit(1) propagates through GitHub Actions as job failure
```

---

## Integration Points (Live API)

All 4 layers are designed to be swapped from mock → live system at a single point:

```python
# MOCK (offline demo)
from evaluation.mock_rag_system import MockRAGSystem
rag = MockRAGSystem()

# LIVE (production DocumentPortal)
from src.document_chat.retrieval import ConversationalRAG
rag = ConversationalRAG(session_id="qa-run")
rag.load_retriever_from_faiss(
    index_path="faiss_index/your_index",
    k=5,
    index_name="index"
)

# Then use identically across all layers:
evaluator = RAGASEvaluator(rag_system=rag)
bias_detector = BiasDetector(rag_system=rag)
fairness_checker = PromptFairnessChecker(rag_system=rag)
```

---

## Dependencies

```
# Base (already in requirements.txt)
langchain, faiss-cpu, fastapi, structlog, pytest

# QA additions (ci_cd_extension/requirements_qa.txt)
numpy>=1.24.0
pytest-cov>=4.1.0
pytest-json-report>=1.5.0

# Optional (auto-detected, not required for demo)
evidently>=0.4.0    ← richer drift reports
ragas>=0.1.0        ← LLM-judge metrics
deepchecks>=0.18.0  ← additional ML validation
pandas>=2.0.0       ← required by evidently
```

---

## File Count Summary

| Layer | Files | Lines of Code |
|---|---|---|
| evaluation/ | 5 files | ~700 |
| monitoring/ | 5 files | ~600 |
| bias_detection/ | 4 files | ~500 |
| ci_cd_extension/ | 3 files | ~200 |
| tests/ (new) | 3 files | ~400 |
| demo_runner.py | 1 file | ~250 |
| **Total new** | **21 files** | **~2,650** |

---

*Built for the Intellectt AI QA Lead interview, May 2026.*  
*System under test: [sunnysavita10/document_portal](https://github.com/sunnysavita10/document_portal)*

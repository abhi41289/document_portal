"""
25+ HR-domain test cases for RAGAS-style evaluation.

JD Requirements covered:
  - LLM testing: prompt evaluation, response validation
  - RAGAS or similar evaluation metrics
  - Test case design covering factual, reasoning, edge cases, out-of-scope

Test case schema:
  id          : unique identifier
  category    : factual | reasoning | edge_case | out_of_scope | bias
  question    : the user query sent to the RAG system
  ground_truth: the correct answer drawn from HR_KNOWLEDGE_BASE
  context     : the relevant document chunk(s) the RAG should have retrieved
  expected_keywords : tokens that MUST appear in a correct answer
  forbidden_keywords: tokens that indicate hallucination
  min_faithfulness  : minimum acceptable faithfulness score (0–1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestCase:
    id: str
    category: str
    question: str
    ground_truth: str
    context: str
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    min_faithfulness: float = 0.75
    notes: str = ""


# ---------------------------------------------------------------------------
# FACTUAL TEST CASES (questions with a direct answer in the policy doc)
# ---------------------------------------------------------------------------

FACTUAL_CASES: list[TestCase] = [
    TestCase(
        id="TC-F-001",
        category="factual",
        question="How many days of annual leave are employees entitled to?",
        ground_truth="Employees are entitled to 18 days of paid annual leave per calendar year.",
        context=(
            "Acme Corp employees are entitled to 18 days of paid annual leave per calendar year. "
            "Leave must be requested at least 7 days in advance via the HR portal. "
            "Unused leave up to 5 days may be carried forward to the next year."
        ),
        expected_keywords=["18", "annual leave", "calendar year"],
        forbidden_keywords=["20 days", "15 days", "30 days", "unlimited"],
        min_faithfulness=0.85,
    ),
    TestCase(
        id="TC-F-002",
        category="factual",
        question="How much advance notice is required to request annual leave?",
        ground_truth="Leave must be requested at least 7 days in advance via the HR portal.",
        context=(
            "Acme Corp employees are entitled to 18 days of paid annual leave per calendar year. "
            "Leave must be requested at least 7 days in advance via the HR portal."
        ),
        expected_keywords=["7 days", "advance", "hr portal"],
        forbidden_keywords=["14 days", "30 days", "no notice"],
        min_faithfulness=0.85,
    ),
    TestCase(
        id="TC-F-003",
        category="factual",
        question="How many days of sick leave does an employee receive?",
        ground_truth="Employees receive 12 days of paid sick leave per year.",
        context=(
            "Employees receive 12 days of paid sick leave per year. "
            "Medical certificates are required for absences exceeding 3 consecutive days."
        ),
        expected_keywords=["12", "sick leave"],
        forbidden_keywords=["10 days", "15 days", "unlimited"],
        min_faithfulness=0.85,
    ),
    TestCase(
        id="TC-F-004",
        category="factual",
        question="When is a medical certificate required for sick leave?",
        ground_truth="Medical certificates are required for absences exceeding 3 consecutive days.",
        context=(
            "Employees receive 12 days of paid sick leave per year. "
            "Medical certificates are required for absences exceeding 3 consecutive days."
        ),
        expected_keywords=["medical certificate", "3 consecutive days"],
        forbidden_keywords=["1 day", "7 days", "never"],
        min_faithfulness=0.80,
    ),
    TestCase(
        id="TC-F-005",
        category="factual",
        question="How long is maternity leave at Acme Corp?",
        ground_truth="Maternity leave is 26 weeks (6 months) of paid leave.",
        context=(
            "Maternity leave is 26 weeks (6 months) of paid leave. "
            "An additional 4 weeks of unpaid leave may be requested."
        ),
        expected_keywords=["26 weeks", "6 months", "paid"],
        forbidden_keywords=["12 weeks", "52 weeks", "8 weeks"],
        min_faithfulness=0.80,
    ),
    TestCase(
        id="TC-F-006",
        category="factual",
        question="What is the paternity leave entitlement?",
        ground_truth="Paternity leave is 2 weeks of paid leave to be taken within 90 days.",
        context=(
            "Paternity leave is 2 weeks of paid leave to be taken within 90 days of birth or adoption."
        ),
        expected_keywords=["2 weeks", "paid"],
        forbidden_keywords=["4 weeks", "1 week", "unpaid"],
        min_faithfulness=0.80,
    ),
    TestCase(
        id="TC-F-007",
        category="factual",
        question="What is the employer 401(k) match percentage?",
        ground_truth="The company offers a 4% employer match on the 401(k) plan.",
        context=(
            "The company offers a 401(k) retirement plan with a 4% employer match. "
            "Employees are eligible after 90 days of employment."
        ),
        expected_keywords=["4%", "employer match", "401"],
        forbidden_keywords=["6%", "3%", "no match"],
        min_faithfulness=0.85,
    ),
    TestCase(
        id="TC-F-008",
        category="factual",
        question="How many remote work days per week are allowed?",
        ground_truth="Eligible employees may work remotely up to 3 days per week.",
        context=(
            "Eligible employees may work remotely up to 3 days per week subject to manager approval."
        ),
        expected_keywords=["3 days", "remote", "manager approval"],
        forbidden_keywords=["5 days", "2 days", "not allowed"],
        min_faithfulness=0.85,
    ),
    TestCase(
        id="TC-F-009",
        category="factual",
        question="What is the annual tuition reimbursement cap?",
        ground_truth="Up to $5,000 per year is reimbursed for job-related education.",
        context=(
            "Acme Corp reimburses up to $5,000 per year for job-related education."
        ),
        expected_keywords=["$5,000", "education"],
        forbidden_keywords=["$10,000", "$2,500", "no reimbursement"],
        min_faithfulness=0.75,
    ),
    TestCase(
        id="TC-F-010",
        category="factual",
        question="When are annual performance reviews conducted?",
        ground_truth="Performance reviews are conducted annually in December.",
        context=(
            "Performance reviews are conducted annually in December. "
            "Mid-year check-ins occur in June."
        ),
        expected_keywords=["december", "performance review", "annually"],
        forbidden_keywords=["march", "june", "quarterly"],
        min_faithfulness=0.80,
    ),
]

# ---------------------------------------------------------------------------
# REASONING TEST CASES (require combining multiple policy clauses)
# ---------------------------------------------------------------------------

REASONING_CASES: list[TestCase] = [
    TestCase(
        id="TC-R-001",
        category="reasoning",
        question=(
            "An employee has been sick for 5 days. What documentation requirements apply "
            "and how many sick days remain if they already used 3 this year?"
        ),
        ground_truth=(
            "A medical certificate is required since the absence exceeds 3 consecutive days. "
            "With 12 days total and 3 already used, 9 sick days remain."
        ),
        context=(
            "Employees receive 12 days of paid sick leave per year. "
            "Medical certificates are required for absences exceeding 3 consecutive days. "
            "Sick leave cannot be carried forward."
        ),
        expected_keywords=["medical certificate", "9", "sick leave"],
        forbidden_keywords=["no documentation needed", "unlimited"],
        min_faithfulness=0.75,
        notes="Tests multi-clause reasoning and arithmetic",
    ),
    TestCase(
        id="TC-R-002",
        category="reasoning",
        question=(
            "Can a new employee who joined 2 months ago enroll in the 401(k) plan? "
            "What about health insurance?"
        ),
        ground_truth=(
            "The employee is not yet eligible for 401(k) since 90 days have not passed. "
            "However, health insurance coverage begins on the first day of employment so they are covered."
        ),
        context=(
            "The company offers a 401(k) retirement plan with a 4% employer match. "
            "Employees are eligible after 90 days of employment. "
            "Acme Corp provides comprehensive health insurance covering medical, dental, and vision. "
            "Coverage begins on the first day of employment."
        ),
        expected_keywords=["90 days", "health insurance", "first day"],
        forbidden_keywords=["immediately eligible for 401k", "not covered"],
        min_faithfulness=0.75,
        notes="Tests cross-policy reasoning",
    ),
    TestCase(
        id="TC-R-003",
        category="reasoning",
        question="If an employee takes full maternity leave, can they also request additional unpaid leave?",
        ground_truth=(
            "Yes. After the 26 weeks of paid maternity leave, an additional 4 weeks of unpaid leave may be requested."
        ),
        context=(
            "Maternity leave is 26 weeks (6 months) of paid leave. "
            "An additional 4 weeks of unpaid leave may be requested."
        ),
        expected_keywords=["4 weeks", "unpaid", "additional"],
        forbidden_keywords=["not allowed", "no additional leave"],
        min_faithfulness=0.80,
    ),
    TestCase(
        id="TC-R-004",
        category="reasoning",
        question=(
            "An employee received $4,000 in tuition reimbursement and left the company "
            "6 months later. What are the financial implications?"
        ),
        ground_truth=(
            "The employee must reimburse the full amount on a pro-rated basis since "
            "they did not remain for the required 1 year after course completion."
        ),
        context=(
            "Acme Corp reimburses up to $5,000 per year for job-related education. "
            "Employees must remain with the company for at least 1 year after course completion "
            "or reimburse the full amount on a pro-rated basis."
        ),
        expected_keywords=["reimburse", "1 year", "pro-rated"],
        forbidden_keywords=["keep the money", "no obligation"],
        min_faithfulness=0.75,
        notes="Tests policy implication reasoning",
    ),
    TestCase(
        id="TC-R-005",
        category="reasoning",
        question=(
            "Can an employee carry forward all 18 days of unused annual leave to next year?"
        ),
        ground_truth=(
            "No. Only up to 5 days of unused annual leave may be carried forward. "
            "The remaining 13 days would be forfeited."
        ),
        context=(
            "Acme Corp employees are entitled to 18 days of paid annual leave per calendar year. "
            "Unused leave up to 5 days may be carried forward to the next year."
        ),
        expected_keywords=["5 days", "carried forward", "not all"],
        forbidden_keywords=["all 18 days", "unlimited carryover"],
        min_faithfulness=0.80,
    ),
]

# ---------------------------------------------------------------------------
# EDGE CASE TEST CASES (boundary conditions, ambiguous questions)
# ---------------------------------------------------------------------------

EDGE_CASES: list[TestCase] = [
    TestCase(
        id="TC-E-001",
        category="edge_case",
        question="Can an employee carry forward sick leave to next year?",
        ground_truth="No. Sick leave cannot be carried forward.",
        context=(
            "Employees receive 12 days of paid sick leave per year. "
            "Sick leave cannot be carried forward."
        ),
        expected_keywords=["cannot", "carried forward"],
        forbidden_keywords=["yes", "can carry", "unlimited"],
        min_faithfulness=0.85,
        notes="Edge case: explicit prohibition clause",
    ),
    TestCase(
        id="TC-E-002",
        category="edge_case",
        question="Does the maternity leave policy apply to adoptive parents?",
        ground_truth="Yes. The maternity leave policy applies to both biological and adoptive parents.",
        context=(
            "Maternity leave is 26 weeks (6 months) of paid leave. "
            "The policy applies to both biological and adoptive parents."
        ),
        expected_keywords=["adoptive", "biological", "both"],
        forbidden_keywords=["biological only", "does not apply"],
        min_faithfulness=0.85,
    ),
    TestCase(
        id="TC-E-003",
        category="edge_case",
        question="What happens if an employee wants to work fully remote?",
        ground_truth=(
            "Fully remote arrangements require VP-level sign-off and a signed remote work agreement."
        ),
        context=(
            "Eligible employees may work remotely up to 3 days per week subject to manager approval. "
            "Fully remote arrangements require VP-level sign-off and a signed remote work agreement."
        ),
        expected_keywords=["vp", "sign-off", "remote work agreement"],
        forbidden_keywords=["not allowed", "manager approval only"],
        min_faithfulness=0.80,
    ),
    TestCase(
        id="TC-E-004",
        category="edge_case",
        question="How quickly must a harassment complaint be investigated?",
        ground_truth="All complaints are investigated within 14 business days.",
        context=(
            "Acme Corp maintains a zero-tolerance policy against workplace harassment. "
            "Incidents must be reported to HR within 30 days. "
            "All complaints are investigated within 14 business days."
        ),
        expected_keywords=["14 business days", "investigated"],
        forbidden_keywords=["30 days", "7 days", "no timeline"],
        min_faithfulness=0.85,
    ),
    TestCase(
        id="TC-E-005",
        category="edge_case",
        question="On what day does health insurance coverage begin?",
        ground_truth="Coverage begins on the first day of employment.",
        context=(
            "Acme Corp provides comprehensive health insurance covering medical, dental, and vision. "
            "Coverage begins on the first day of employment."
        ),
        expected_keywords=["first day", "employment"],
        forbidden_keywords=["90 days", "probation", "after 30 days"],
        min_faithfulness=0.85,
    ),
    TestCase(
        id="TC-E-006",
        category="edge_case",
        question="Is there a dress code for client meetings?",
        ground_truth=(
            "Yes. Client-facing meetings require formal business attire even if it's a Friday."
        ),
        context=(
            "Acme Corp follows a business casual dress code Monday through Thursday. "
            "Fridays are designated casual days. Client-facing meetings require formal business attire."
        ),
        expected_keywords=["formal", "client", "business attire"],
        forbidden_keywords=["casual", "no requirement", "dress code does not apply"],
        min_faithfulness=0.80,
    ),
]

# ---------------------------------------------------------------------------
# OUT-OF-SCOPE TEST CASES (questions outside the HR document domain)
# ---------------------------------------------------------------------------

OUT_OF_SCOPE_CASES: list[TestCase] = [
    TestCase(
        id="TC-OOS-001",
        category="out_of_scope",
        question="What is the current stock price of Acme Corp?",
        ground_truth="This information is not available in the provided HR policy documents.",
        context="",
        expected_keywords=["not available", "policy documents"],
        forbidden_keywords=["$", "stock", "price", "trading"],
        min_faithfulness=0.90,
        notes="Financial data not in HR doc — RAG must not hallucinate",
    ),
    TestCase(
        id="TC-OOS-002",
        category="out_of_scope",
        question="Who is the current CEO of Acme Corp?",
        ground_truth="This information is not available in the provided HR policy documents.",
        context="",
        expected_keywords=["not available", "policy documents"],
        forbidden_keywords=["ceo", "chief executive", "named"],
        min_faithfulness=0.90,
        notes="Leadership info not in HR doc",
    ),
    TestCase(
        id="TC-OOS-003",
        category="out_of_scope",
        question="What is the recipe for Acme Corp's office birthday cake?",
        ground_truth="This information is not available in the provided HR policy documents.",
        context="",
        expected_keywords=["not available"],
        forbidden_keywords=["flour", "sugar", "recipe"],
        min_faithfulness=0.90,
        notes="Completely irrelevant query",
    ),
    TestCase(
        id="TC-OOS-004",
        category="out_of_scope",
        question="What is the weather forecast for the Acme Corp headquarters?",
        ground_truth="This information is not available in the provided HR policy documents.",
        context="",
        expected_keywords=["not available"],
        forbidden_keywords=["sunny", "rain", "forecast", "temperature"],
        min_faithfulness=0.90,
    ),
    TestCase(
        id="TC-OOS-005",
        category="out_of_scope",
        question="Tell me about the history of workplace rights movements globally.",
        ground_truth="This information is not available in the provided HR policy documents.",
        context="",
        expected_keywords=["not available"],
        forbidden_keywords=["labor movement", "1900", "union", "strike"],
        min_faithfulness=0.85,
        notes="Tests that model doesn't use external knowledge",
    ),
]

# ---------------------------------------------------------------------------
# CONSISTENCY / PARAPHRASE TEST CASES (used by bias_detection layer)
# ---------------------------------------------------------------------------

PARAPHRASE_GROUPS: list[dict] = [
    {
        "group_id": "PG-001",
        "topic": "annual_leave_entitlement",
        "variants": [
            "How many days of annual leave do I get?",
            "What is the paid vacation entitlement?",
            "How much time off per year is allowed?",
            "How many holiday days does the company give?",
            "What is the annual leave policy at Acme Corp?",
        ],
        "expected_core_answer": "18 days",
    },
    {
        "group_id": "PG-002",
        "topic": "401k_eligibility",
        "variants": [
            "When can I join the retirement plan?",
            "When am I eligible for the 401k?",
            "How long before I can contribute to the pension?",
            "What is the waiting period for retirement benefits?",
            "When does the company match my 401k contributions?",
        ],
        "expected_core_answer": "90 days",
    },
    {
        "group_id": "PG-003",
        "topic": "harassment_reporting",
        "variants": [
            "How do I report workplace harassment?",
            "What should I do if I'm being bullied at work?",
            "Where do I file a harassment complaint?",
            "Who do I contact about workplace misconduct?",
            "What is the process for reporting a colleague's inappropriate behavior?",
        ],
        "expected_core_answer": "hr",
    },
]

# ---------------------------------------------------------------------------
# Master collection
# ---------------------------------------------------------------------------

HR_TEST_CASES: list[TestCase] = (
    FACTUAL_CASES
    + REASONING_CASES
    + EDGE_CASES
    + OUT_OF_SCOPE_CASES
)

CATEGORY_COUNTS = {
    "factual": len(FACTUAL_CASES),
    "reasoning": len(REASONING_CASES),
    "edge_case": len(EDGE_CASES),
    "out_of_scope": len(OUT_OF_SCOPE_CASES),
    "total": len(HR_TEST_CASES),
}

if __name__ == "__main__":
    print(f"Total test cases: {CATEGORY_COUNTS['total']}")
    for cat, count in CATEGORY_COUNTS.items():
        if cat != "total":
            print(f"  {cat}: {count}")

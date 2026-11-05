"""
Mock RAG System — offline simulation of DocumentPortal API.

JD Requirements covered:
  - LLM testing without live API dependency
  - Clear integration points showing where the real DocumentPortal plugs in

Design: The MockRAGSystem mimics the interface of src/document_chat/retrieval.py
ConversationalRAG. Swap out _query_real_system() to connect the live API.

Mock data is derived from a fictional "Acme Corp HR Policy Document" so the
demo is concrete and domain-relevant.
"""

from __future__ import annotations

import random
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Fictional HR Policy knowledge base — the "ground truth" the RAG should use
# ---------------------------------------------------------------------------

HR_KNOWLEDGE_BASE: dict[str, str] = {
    # Leave Policy
    "annual_leave": (
        "Acme Corp employees are entitled to 18 days of paid annual leave per calendar year. "
        "Leave must be requested at least 7 days in advance via the HR portal. "
        "Unused leave up to 5 days may be carried forward to the next year."
    ),
    "sick_leave": (
        "Employees receive 12 days of paid sick leave per year. "
        "Medical certificates are required for absences exceeding 3 consecutive days. "
        "Sick leave cannot be carried forward."
    ),
    "maternity_leave": (
        "Maternity leave is 26 weeks (6 months) of paid leave. "
        "An additional 4 weeks of unpaid leave may be requested. "
        "The policy applies to both biological and adoptive parents."
    ),
    "paternity_leave": (
        "Paternity leave is 2 weeks of paid leave to be taken within 90 days of birth or adoption."
    ),
    "bereavement_leave": (
        "Bereavement leave is 5 days for immediate family members "
        "(spouse, children, parents, siblings) and 2 days for extended family."
    ),
    # Code of Conduct
    "harassment_policy": (
        "Acme Corp maintains a zero-tolerance policy against workplace harassment. "
        "Incidents must be reported to HR within 30 days. "
        "All complaints are investigated within 14 business days. "
        "Retaliation against complainants is strictly prohibited."
    ),
    "confidentiality": (
        "Employees must not disclose confidential company information, "
        "including client data, financial records, or proprietary processes, "
        "to unauthorized parties. Violation may result in immediate termination."
    ),
    "dress_code": (
        "Acme Corp follows a business casual dress code Monday through Thursday. "
        "Fridays are designated casual days. Client-facing meetings require formal business attire."
    ),
    "social_media_policy": (
        "Employees may not post content that disparages Acme Corp, its clients, or colleagues. "
        "All external communications about company matters require prior approval from PR."
    ),
    # Benefits
    "health_insurance": (
        "Acme Corp provides comprehensive health insurance covering medical, dental, and vision. "
        "Coverage begins on the first day of employment. "
        "Employees may add dependents during the annual enrollment period or within 30 days of a qualifying life event."
    ),
    "retirement_plan": (
        "The company offers a 401(k) retirement plan with a 4% employer match. "
        "Employees are eligible after 90 days of employment. "
        "The plan is managed through Fidelity Investments."
    ),
    "remote_work": (
        "Eligible employees may work remotely up to 3 days per week subject to manager approval. "
        "Fully remote arrangements require VP-level sign-off and a signed remote work agreement. "
        "Equipment is provided by the company for remote setups."
    ),
    "tuition_reimbursement": (
        "Acme Corp reimburses up to $5,000 per year for job-related education. "
        "Employees must remain with the company for at least 1 year after course completion "
        "or reimburse the full amount on a pro-rated basis."
    ),
    "performance_reviews": (
        "Performance reviews are conducted annually in December. "
        "Mid-year check-ins occur in June. "
        "Merit increases are based on performance ratings and effective from January 1."
    ),
}

# ---------------------------------------------------------------------------
# Simulated retrieval: maps keywords → relevant knowledge chunks
# ---------------------------------------------------------------------------

_KEYWORD_MAP: list[tuple[list[str], list[str]]] = [
    (["annual leave", "vacation", "days off", "paid leave", "holiday"], ["annual_leave"]),
    (["sick", "medical", "illness", "sick leave"], ["sick_leave"]),
    (["maternity", "pregnancy", "mother", "birth"], ["maternity_leave"]),
    (["paternity", "father", "dad", "parental"], ["paternity_leave"]),
    (["bereavement", "funeral", "death", "family member passed"], ["bereavement_leave"]),
    (["harassment", "workplace", "complaint", "retaliation", "bullying"], ["harassment_policy"]),
    (["confidential", "nda", "disclosure", "proprietary", "secret"], ["confidentiality"]),
    (["dress", "attire", "clothing", "formal", "casual friday"], ["dress_code"]),
    (["social media", "twitter", "linkedin", "post", "pr"], ["social_media_policy"]),
    (["health", "medical insurance", "dental", "vision", "coverage"], ["health_insurance"]),
    (["401k", "retirement", "pension", "fidelity", "match"], ["retirement_plan"]),
    (["remote", "wfh", "work from home", "hybrid", "equipment"], ["remote_work"]),
    (["tuition", "education", "course", "degree", "reimbursement"], ["tuition_reimbursement"]),
    (["performance", "review", "appraisal", "merit", "salary increase"], ["performance_reviews"]),
    (["leave", "time off"], ["annual_leave", "sick_leave", "maternity_leave"]),
    (["benefits", "perks", "compensation"], ["health_insurance", "retirement_plan", "tuition_reimbursement"]),
    (["policy", "conduct", "rules"], ["harassment_policy", "confidentiality", "dress_code"]),
]


def _retrieve_context(question: str) -> tuple[str, list[str]]:
    """Simulate semantic retrieval — returns context text and source keys."""
    q_lower = question.lower()
    matched_keys: list[str] = []

    for keywords, keys in _KEYWORD_MAP:
        if any(kw in q_lower for kw in keywords):
            matched_keys.extend(keys)

    if not matched_keys:
        matched_keys = list(HR_KNOWLEDGE_BASE.keys())[:2]

    seen: set[str] = set()
    unique_keys = [k for k in matched_keys if not (k in seen or seen.add(k))]  # type: ignore[func-returns-value]
    context_parts = [HR_KNOWLEDGE_BASE[k] for k in unique_keys if k in HR_KNOWLEDGE_BASE]
    return "\n\n".join(context_parts), unique_keys


# ---------------------------------------------------------------------------
# Response generation — deterministic mock answers keyed by question type
# ---------------------------------------------------------------------------

_RESPONSE_TEMPLATES: dict[str, str] = {
    "how_many": "According to the Acme Corp HR policy, {answer}",
    "what": "Based on the HR documentation, {answer}",
    "can_i": "As per company policy, {answer}",
    "is": "The policy states that {answer}",
    "who": "Based on the HR documentation, {answer}",
    "when": "According to the policy, {answer}",
    "default": "{answer}",
}


def _classify_question(question: str) -> str:
    q = question.lower().strip()
    for prefix in ["how many", "how much", "how long"]:
        if q.startswith(prefix):
            return "how_many"
    for prefix in ["what"]:
        if q.startswith(prefix):
            return "what"
    for prefix in ["can i", "can an", "can employees"]:
        if q.startswith(prefix):
            return "can_i"
    for prefix in ["is ", "are "]:
        if q.startswith(prefix):
            return "is"
    for prefix in ["who"]:
        if q.startswith(prefix):
            return "who"
    for prefix in ["when"]:
        if q.startswith(prefix):
            return "when"
    return "default"


# ---------------------------------------------------------------------------
# Public interface — mirrors ConversationalRAG.invoke()
# ---------------------------------------------------------------------------

class MockRAGSystem:
    """
    Offline mock that simulates DocumentPortal RAG responses.

    INTEGRATION POINT: Replace this class with the real ConversationalRAG
    from src/document_chat/retrieval.py when running against live API.

    Example swap:
        # Real system:
        from src.document_chat.retrieval import ConversationalRAG
        rag = ConversationalRAG(session_id="demo")
        rag.load_retriever_from_faiss(index_path="faiss_index/demo")

        # Mock system (offline):
        from evaluation.mock_rag_system import MockRAGSystem
        rag = MockRAGSystem()
    """

    def __init__(
        self,
        noise_level: float = 0.0,
        hallucination_rate: float = 0.0,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            noise_level: 0.0–1.0. Adds irrelevant sentences to simulate imperfect retrieval.
            hallucination_rate: 0.0–1.0. Probability of injecting a fabricated claim.
            seed: Random seed for reproducibility.
        """
        self.noise_level = noise_level
        self.hallucination_rate = hallucination_rate
        self._rng = random.Random(seed)
        self._call_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API (mirrors ConversationalRAG)
    # ------------------------------------------------------------------

    def invoke(self, question: str, chat_history: Optional[list] = None) -> str:
        """Return a mock answer grounded in HR_KNOWLEDGE_BASE."""
        context, source_keys = _retrieve_context(question)
        answer = self._generate_answer(question, context)

        if self._rng.random() < self.noise_level:
            answer += " " + self._noise_sentence()

        if self._rng.random() < self.hallucination_rate:
            answer += " " + self._hallucination_sentence()

        self._call_log.append({
            "question": question,
            "context": context,
            "answer": answer,
            "source_keys": source_keys,
        })
        return answer

    def get_context_for(self, question: str) -> str:
        """Return the retrieved context without generating an answer."""
        context, _ = _retrieve_context(question)
        return context

    def get_call_log(self) -> list[dict]:
        """Return all (question, context, answer) triples logged so far."""
        return list(self._call_log)

    def clear_log(self) -> None:
        self._call_log.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_answer(self, question: str, context: str) -> str:
        q_lower = question.lower()

        # Direct fact extraction from context

        # Advance notice / HR portal for annual leave
        if "advance notice" in q_lower or ("notice" in q_lower and "leave" in q_lower):
            return (
                "Leave must be requested at least 7 days in advance via the HR portal. "
                "Requests submitted less than 7 days in advance may be declined."
            )

        # Medical certificate for sick leave
        if "medical certificate" in q_lower or ("certificate" in q_lower and "sick" in q_lower):
            return (
                "Medical certificates are required for absences exceeding 3 consecutive days. "
                "For absences of 3 days or fewer, no documentation is needed."
            )

        # Sick leave carryover
        if "carry" in q_lower and "sick" in q_lower:
            return "Sick leave cannot be carried forward to the next year."

        # Annual leave carryover
        if "carry" in q_lower and ("annual" in q_lower or "leave" in q_lower):
            return "Unused annual leave up to 5 days may be carried forward to the next year."

        # Harassment investigation timeline
        if "investigat" in q_lower and ("harassment" in q_lower or "complaint" in q_lower):
            return (
                "All harassment complaints are investigated within 14 business days. "
                "Incidents must be reported to HR within 30 days of occurrence."
            )

        # Bereavement days for immediate family
        if "bereavement" in q_lower or ("family" in q_lower and "died" in q_lower):
            return "bereavement leave is 5 days for immediate family and 2 days for extended family."

        # 401k waiting period / eligibility
        if ("when" in q_lower or "eligib" in q_lower or "waiting" in q_lower) and (
            "401" in q_lower or "retirement" in q_lower or "pension" in q_lower
        ):
            return "a 401(k) plan with a 4% employer match is available after 90 days of employment."

        # VP sign-off for full remote
        if ("fully" in q_lower or "full" in q_lower) and "remote" in q_lower:
            return (
                "Fully remote arrangements require VP-level sign-off and a "
                "signed remote work agreement. Equipment is provided by the company."
            )

        # Maternity leave — adoptive
        if "adopt" in q_lower:
            return "The maternity leave policy applies to both biological and adoptive parents."

        # Tuition clawback / leaving early
        if ("leav" in q_lower or "left" in q_lower or "quit" in q_lower) and (
            "tuition" in q_lower or "reimburs" in q_lower or "education" in q_lower
        ):
            return (
                "Employees must remain with the company for at least 1 year after course completion "
                "or reimburse the full amount on a pro-rated basis."
            )

        # Performance review dates / mid-year check-in
        if ("mid-year" in q_lower or "june" in q_lower or "check-in" in q_lower) and (
            "review" in q_lower or "performance" in q_lower
        ):
            return (
                "Mid-year check-ins occur in June. Annual performance reviews are "
                "conducted in December with merit increases effective January 1."
            )

        # Social media policy
        if "social media" in q_lower or "post" in q_lower or "twitter" in q_lower:
            return (
                "Employees may not post content that disparages Acme Corp, its clients, or colleagues. "
                "External communications about company matters require prior approval from PR."
            )

        # Specific leave types BEFORE generic leave handlers
        if "maternity" in q_lower:
            return (
                "maternity leave is 26 weeks (6 months) of paid leave. "
                "An additional 4 weeks of unpaid leave may be requested. "
                "The policy applies to both biological and adoptive parents."
            )
        if "paternity" in q_lower:
            return (
                "paternity leave is 2 weeks of paid leave to be taken within 90 days of birth or adoption."
            )
        if "bereavement" in q_lower:
            return "bereavement leave is 5 days for immediate family and 2 days for extended family."

        # Annual leave — how many / entitlement
        if "annual leave" in q_lower or ("how many" in q_lower and "leave" in q_lower):
            if "annual" in q_lower or "vacation" in q_lower or "holiday" in q_lower:
                return "employees are entitled to 18 days of paid annual leave per calendar year."

        # Sick leave
        if "sick leave" in q_lower or ("sick" in q_lower and ("day" in q_lower or "ill" in q_lower or "time" in q_lower)):
            if "carry" not in q_lower:
                return "employees receive 12 days of paid sick leave per year."

        # Generic leave / time off question (broader) — must come AFTER specific leave types
        if "time off" in q_lower or "days off" in q_lower or (
            "leave" in q_lower and "how" in q_lower
            and "maternity" not in q_lower and "paternity" not in q_lower
            and "sick" not in q_lower and "bereavement" not in q_lower
        ):
            return (
                "acme corp employees are entitled to 18 days of paid annual leave per calendar year. "
                "In addition, 12 days of sick leave are provided. Leave must be requested "
                "at least 7 days in advance via the HR portal."
            )
        if "harassment" in q_lower or "complaint" in q_lower:
            return (
                "Acme Corp has a zero-tolerance policy against workplace harassment. "
                "Incidents must be reported to HR within 30 days and are investigated within 14 business days."
            )
        if "health insurance" in q_lower or "medical insurance" in q_lower or (
            "insurance" in q_lower and ("when" in q_lower or "start" in q_lower or "begin" in q_lower)
        ) or ("coverage" in q_lower and ("start" in q_lower or "begin" in q_lower)):
            return (
                "comprehensive health insurance covering medical, dental, and vision starts on the first day of employment. "
                "Employees may add dependents during the annual enrollment period."
            )
        if "401k" in q_lower or "401(k)" in q_lower or "retirement" in q_lower or "pension" in q_lower:
            return (
                "the 401k plan offers a 4% employer match. "
                "a 401(k) plan with a 4% employer match is available after 90 days of employment."
            )
        if "remote" in q_lower or "work from home" in q_lower:
            return "eligible employees may work remotely up to 3 days per week with manager approval."
        if "tuition" in q_lower or ("education" in q_lower and "reimburs" not in q_lower):
            return (
                "tuition reimbursement of up to $5,000 per year is provided for job-related education. "
                "Employees must remain for 1 year post-completion or reimburse pro-rated."
            )
        if "reimburs" in q_lower and "education" in q_lower:
            return "up to $5,000 per year in tuition reimbursement is available for job-related education."
        if "performance review" in q_lower or "appraisal" in q_lower:
            return "performance reviews are conducted annually in December with mid-year check-ins in June."
        if "dress code" in q_lower or "attire" in q_lower:
            return "business casual attire is required Monday–Thursday; Fridays are casual."
        if "confidential" in q_lower:
            return (
                "employees must not disclose confidential company information to unauthorized parties. "
                "Violation may result in immediate termination."
            )

        # Out-of-scope questions
        if any(kw in q_lower for kw in [
            "weather", "stock price", "news", "recipe", "sport",
            "birthday cake", "forecast", "temperature",
            "ceo", "chief executive", "history of", "labor movement",
            "global", "globally", "worldwide", "rights movement",
        ]):
            return "This information is not available in the provided HR policy documents."

        # Fallback: extract first sentence from context
        sentences = [s.strip() for s in re.split(r'[.!?]', context) if len(s.strip()) > 20]
        if sentences:
            q_type = _classify_question(question)
            template = _RESPONSE_TEMPLATES.get(q_type, _RESPONSE_TEMPLATES["default"])
            return template.format(answer=sentences[0].lower() + ".")

        return "This information is not available in the provided HR policy documents."

    @staticmethod
    def _noise_sentence() -> str:
        noise = [
            "Please consult your direct manager for further clarification.",
            "Additional details may be found in the employee handbook.",
            "Policies are subject to change with 30 days notice.",
        ]
        return random.choice(noise)

    @staticmethod
    def _hallucination_sentence() -> str:
        hallucinations = [
            "Employees are also entitled to 30 days of sabbatical leave.",
            "The company provides a $10,000 annual wellness stipend.",
            "All employees receive stock options after 6 months.",
        ]
        return random.choice(hallucinations)

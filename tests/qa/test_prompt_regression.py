"""Prompt version comparison and regression checks."""

from __future__ import annotations

import pytest

from prompt.prompt_library import PROMPT_VERSIONS, rag_generation_prompt
from tests.qa.datasets import PROMPT_REGRESSION_CASES
from tests.qa.helpers import make_llm_response
from tests.qa.metrics import exact_match, f1_score


@pytest.mark.integration
@pytest.mark.regression
def test_prompt_version_comparison_with_mock_llm(qa_prompt_cases):
    v1_outputs = [
        "The revenue is $10 million.",
        "Bob Smith founded the company.",
        "The company was founded in 2012.",
    ]
    v2_outputs = [
        "Not available in provided documents.",
        "Alice Johnson founded TechCorp.",
        "The documents conflict on the founding year.",
    ]

    class MockLLM:
        def __init__(self, outputs):
            self.outputs = list(outputs)

        def invoke(self, prompt):
            return make_llm_response(self.outputs.pop(0))

    v1_llm = MockLLM(v1_outputs)
    v2_llm = MockLLM(v2_outputs)

    v1_scores = []
    v2_scores = []

    for case in qa_prompt_cases:
        v1_prompt = f"Context: {case.context}\nQuestion: {case.question}\nAnswer:"
        v2_prompt = rag_generation_prompt.format(context=case.context, question=case.question)

        v1_answer = v1_llm.invoke(v1_prompt).content.lower()
        v2_answer = v2_llm.invoke(v2_prompt).content.lower()

        v1_supported = any(keyword in v1_answer for keyword in case.expected_keywords)
        v2_supported = any(keyword in v2_answer for keyword in case.expected_keywords)

        v1_penalty = any(keyword in v1_answer for keyword in case.forbidden_keywords)
        v2_penalty = any(keyword in v2_answer for keyword in case.forbidden_keywords)

        v1_scores.append(1.0 if v1_supported and not v1_penalty else 0.0)
        v2_scores.append(1.0 if v2_supported and not v2_penalty else 0.0)

    assert sum(v2_scores) >= sum(v1_scores)
    assert sum(v2_scores) >= 2


@pytest.mark.regression
def test_prompt_regression_scoring_rules():
    assert exact_match("Not available", "not available") == 1.0
    assert f1_score("alice johnson founded", "alice johnson founded techcorp") > 0.5


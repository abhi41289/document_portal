"""Retrieval quality regression tests."""

from __future__ import annotations

import pytest

from tests.qa.datasets import GOLDEN_DOCUMENTS
from tests.qa.helpers import build_retriever
from tests.qa.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


@pytest.mark.retrieval
@pytest.mark.regression
def test_retrieval_precision_recall_and_mrr(qa_retrieval_cases, qa_documents):
    vocabulary = [
        "ceo",
        "alice",
        "johnson",
        "founded",
        "revenue",
        "q3",
        "q2",
        "q4",
        "planned",
        "roadmap",
    ]
    retriever = build_retriever(vocabulary=vocabulary, chunk_size=240, overlap=30)

    for document in qa_documents:
        retriever.index_document(document["id"], document["content"], document["metadata"])

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    for case in qa_retrieval_cases:
        results = retriever.retrieve(case.question, top_k=case.top_k)
        retrieved_ids = [result.chunk.source_document_id for result in results]

        precision_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, case.top_k))
        recall_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, case.top_k))

        rank = next((idx + 1 for idx, doc_id in enumerate(retrieved_ids) if doc_id in case.relevant_doc_ids), 0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        assert case.relevant_doc_ids[0] in retrieved_ids

    assert sum(precision_scores) / len(precision_scores) >= 0.5
    assert sum(recall_scores) / len(recall_scores) >= 0.5
    assert mean_reciprocal_rank([1 if rank > 0 else 0 for rank in [1, 2, 1]]) > 0.0
    assert sum(reciprocal_ranks) / len(reciprocal_ranks) > 0.0


@pytest.mark.retrieval
def test_retrieval_returns_ranked_results(qa_documents):
    retriever = build_retriever(vocabulary=["ceo", "revenue", "q2", "q3", "roadmap"], chunk_size=200)
    for document in qa_documents:
        retriever.index_document(document["id"], document["content"], document["metadata"])

    results = retriever.retrieve("Who is the CEO?", top_k=3)

    assert results
    assert results[0].rank == 1
    assert results[0].similarity_score >= results[-1].similarity_score

"""Reusable helpers for deterministic QA tests."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

from src.document_chat.retrieval_layer import (
    DocumentRetriever,
    InMemoryVectorStore,
    SlidingWindowChunker,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


class SimpleKeywordEmbedder:
    """Deterministic bag-of-keywords embedder for tests."""

    def __init__(self, vocabulary: Sequence[str]):
        self.vocabulary = [term.lower() for term in vocabulary]
        self.index = {term: idx for idx, term in enumerate(self.vocabulary)}

    def _vectorize(self, text: str) -> List[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * len(self.vocabulary)
        for term, idx in self.index.items():
            vector[idx] = float(counts.get(term, 0))
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_text(self, text: str) -> List[float]:
        return self._vectorize(text)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vectorize(text) for text in texts]


def build_retriever(vocabulary: Sequence[str], chunk_size: int = 220, overlap: int = 30) -> DocumentRetriever:
    embedder = SimpleKeywordEmbedder(vocabulary=vocabulary)
    vector_store = InMemoryVectorStore()
    chunker = SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap)
    return DocumentRetriever(embedder=embedder, vector_store=vector_store, chunking_strategy=chunker)


def make_llm_response(content: str, **metadata: Any) -> SimpleNamespace:
    return SimpleNamespace(content=content, metadata=metadata or {})


def make_mock_llm(responses: Iterable[str]):
    queue = list(responses)

    class _MockLLM:
        def __init__(self):
            self.calls: List[str] = []

        def invoke(self, prompt: str):
            self.calls.append(prompt)
            content = queue.pop(0) if queue else "Not available in provided documents."
            return make_llm_response(content)

    return _MockLLM()


def create_controlled_sqlite_db() -> sqlite3.Connection:
    from tests.qa.datasets import CONTROLLED_ROWS

    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE invoices (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            total REAL NOT NULL,
            status TEXT NOT NULL
        )
        """
    )

    cursor.executemany(
        "INSERT INTO customers (id, name, region, status) VALUES (:id, :name, :region, :status)",
        CONTROLLED_ROWS["customers"],
    )
    cursor.executemany(
        "INSERT INTO orders (id, customer_id, amount, created_at) VALUES (:id, :customer_id, :amount, :created_at)",
        CONTROLLED_ROWS["orders"],
    )
    cursor.executemany(
        "INSERT INTO invoices (id, customer_id, total, status) VALUES (:id, :customer_id, :total, :status)",
        CONTROLLED_ROWS["invoices"],
    )
    connection.commit()
    return connection


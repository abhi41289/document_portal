"""Minimal smoke tests for the FastAPI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

try:
    from api.main import app
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"Skipping API smoke tests because an optional dependency is missing: {exc}", allow_module_level=True)


client = TestClient(app)


@pytest.mark.unit
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text


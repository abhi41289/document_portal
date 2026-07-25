"""Basic route smoke tests for the FastAPI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

try:
    from api.main import app
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"Skipping route smoke tests because an optional dependency is missing: {exc}",
        allow_module_level=True,
    )


client = TestClient(app)


@pytest.mark.unit
def test_health_route():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "document-portal"}


@pytest.mark.unit
def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text


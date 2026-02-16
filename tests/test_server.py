"""Tests for the FastAPI server endpoints."""

import json
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from schism.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Schism" in resp.text


def test_api_models(client):
    with patch("schism.server.list_available_models") as mock:
        mock.return_value = [
            {"name": "gemma-2-2b", "hf_id": "google/gemma-2-2b", "has_sae": True, "loaded": False},
        ]
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["name"] == "gemma-2-2b"


def test_api_features(client):
    resp = client.get("/api/features")
    assert resp.status_code == 200
    data = resp.json()
    assert "creativity" in data
    assert "description" in data["creativity"]


def test_api_presets_list(client):
    resp = client.get("/api/presets")
    assert resp.status_code == 200
    data = resp.json()
    names = [p["name"] for p in data]
    assert "Creative" in names


def test_api_presets_create(client, tmp_path, monkeypatch):
    test_presets = tmp_path / "presets"
    test_presets.mkdir()
    monkeypatch.setattr("schism.presets.manager.PRESETS_DIR", test_presets)

    resp = client.post("/api/presets", json={
        "name": "Test API Preset",
        "description": "Created via API",
        "sliders": {"humor": 0.5},
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_delete_nonexistent_preset(client, tmp_path, monkeypatch):
    test_presets = tmp_path / "presets"
    test_presets.mkdir()
    monkeypatch.setattr("schism.presets.manager.PRESETS_DIR", test_presets)

    resp = client.delete("/api/presets/nonexistent")
    assert resp.status_code == 404


def test_static_css(client):
    resp = client.get("/static/css/style.css")
    assert resp.status_code == 200
    assert "text/css" in resp.headers["content-type"]


def test_static_js(client):
    resp = client.get("/static/js/app.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]

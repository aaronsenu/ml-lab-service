"""
test_app.py
-----------
Unit tests for the Flask API endpoints.
Run with: pytest test_app.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock

# Patch the heavy model loading before importing app
mock_prediction = [{"label": "POSITIVE", "score": 0.99}]

with patch("transformers.pipeline", return_value=lambda text: mock_prediction):
    from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status(client):
    data = json.loads(response := client.get("/health").data)
    assert data["status"] == "ok"


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_returns_200_with_valid_input(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "I love this!"}),
        content_type="application/json"
    )
    assert response.status_code == 200


def test_predict_response_contains_required_fields(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "Great product"}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert "input" in data
    assert "label" in data
    assert "score" in data


def test_predict_returns_400_with_no_body(client):
    response = client.post("/predict", content_type="application/json")
    assert response.status_code == 400


def test_predict_returns_400_with_missing_text_key(client):
    response = client.post(
        "/predict",
        data=json.dumps({"wrong_key": "hello"}),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_predict_echoes_input_text(client):
    text = "This is a test sentence."
    response = client.post(
        "/predict",
        data=json.dumps({"text": text}),
        content_type="application/json"
    )
    data = json.loads(response.data)
    assert data["input"] == text

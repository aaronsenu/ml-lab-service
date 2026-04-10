"""
test_app.py
-----------
Unit tests for the Flask API endpoints.
Run with: pytest test_app.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock

mock_prediction = [{"label": "POSITIVE", "score": 0.99}]
mock_classifier = MagicMock(return_value=mock_prediction)

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "mysecretkey123"
}


@pytest.fixture
def client():
    with patch("transformers.pipeline", return_value=mock_classifier):
        import importlib
        import app as app_module
        importlib.reload(app_module)
        app_module.classifier = mock_classifier
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status(client):
    data = json.loads(client.get("/health").data)
    assert data["status"] == "ok"


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_returns_200_with_valid_input(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "I love this!"}),
        headers=HEADERS
    )
    assert response.status_code == 200


def test_predict_response_contains_required_fields(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "Great product"}),
        headers=HEADERS
    )
    data = json.loads(response.data)
    assert "input" in data
    assert "label" in data
    assert "score" in data


def test_predict_returns_400_with_no_body(client):
    response = client.post("/predict", headers=HEADERS)
    assert response.status_code == 400


def test_predict_returns_400_with_missing_text_key(client):
    response = client.post(
        "/predict",
        data=json.dumps({"wrong_key": "hello"}),
        headers=HEADERS
    )
    assert response.status_code == 400


def test_predict_echoes_input_text(client):
    text = "This is a test sentence."
    response = client.post(
        "/predict",
        data=json.dumps({"text": text}),
        headers=HEADERS
    )
    data = json.loads(response.data)
    assert data["input"] == text


def test_predict_returns_401_without_api_key(client):
    response = client.post(
        "/predict",
        data=json.dumps({"text": "hello"}),
        content_type="application/json"
    )
    assert response.status_code == 401
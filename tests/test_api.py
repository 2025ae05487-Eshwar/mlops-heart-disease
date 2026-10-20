
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint():
    response = client.get("/ready")

    assert response.status_code == 200


def test_predict_rejects_empty_payload():
    response = client.post("/predict", json={})

    # Expected if the endpoint requires patient features.
    assert response.status_code in (400, 422)


def test_predict_rejects_invalid_payload_type():
    response = client.post("/predict", json=[])

    assert response.status_code in (400, 422)

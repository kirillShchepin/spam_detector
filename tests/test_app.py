"""Тесты для API спам-детектора."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Тест корневого эндпоинта."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "error")
    assert "model_loaded" in data


def test_predict_spam():
    """Тест классификации спам-сообщения."""
    response = client.post(
        "/predict",
        json={"text": "Win a free prize now!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] in ("spam", "ham")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_ham():
    """Тест классификации обычного сообщения."""
    response = client.post(
        "/predict",
        json={"text": "Hello, how are you?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] in ("spam", "ham")
    assert 0.0 <= data["confidence"] <= 1.0

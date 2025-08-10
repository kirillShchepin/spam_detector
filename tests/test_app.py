from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_root():
    """Тест корневого эндпоинта."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] in ("ok", "error")


def test_predict_spam():
    """Тест классификации спам-сообщения."""
    response = client.post(
        "/predict",
        json={"text": "Win a free prize now!"}
    )
    assert response.status_code == 200
    assert response.json()["result"] in ["spam", "ham"]


def test_predict_ham():
    """Тест классификации обычного сообщения."""
    response = client.post(
        "/predict",
        json={"text": "Hello, how are you?"}
    )
    assert response.status_code == 200
    assert response.json()["result"] in ["spam", "ham"]

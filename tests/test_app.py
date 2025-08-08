import sys
import os

# Добавляем путь к корню проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_spam():
    """Тестируем, что текст со спам-словами определится как spam"""
    response = client.post(
        "/predict",
        json={"text": "Win a free iPhone!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["result"] in ["spam", "ham"]


def test_predict_ham():
    """Тестируем, что обычный текст определится как ham"""
    response = client.post(
        "/predict",
        json={"text": "Hello, how are you?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "ham"

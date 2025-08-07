import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_spam():
    response = client.post("/predict", json={"text": "Win a free iPhone!"})
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "score" in data
    assert isinstance(data["score"], float)
    # Проверяем возможные метки, которые возвращает модель
    assert data["result"] in ["HAM", "SPAM", "LABEL_0", "LABEL_1"]

def test_predict_ham():
    response = client.post("/predict", json={"text": "Hello, how are you?"})
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "score" in data
    assert isinstance(data["score"], float)
    # Проверяем возможные метки, которые возвращает модель
    assert data["result"] in ["HAM", "SPAM", "LABEL_0", "LABEL_1"]

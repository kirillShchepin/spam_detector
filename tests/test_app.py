import sys
import os

# Добавляем путь к корню проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_spam():
    response = client.post("/predict", json={"text": "Win a free iPhone!"})
    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] in ["HAM", "SPAM"]

def test_predict_ham():
    response = client.post("/predict", json={"text": "Hello, how are you?"})
    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] in ["HAM", "SPAM"]

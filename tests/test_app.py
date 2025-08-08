import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

client = TestClient(app)


def test_predict_spam():
    response = client.post(
        "/predict",
        json={"text": "Win a free iPhone!"}
    )
    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] in ["spam", "ham"]
    assert isinstance(response.json()["confidence"], float)


def test_predict_ham():
    response = client.post(
        "/predict",
        json={"text": "Hello, how are you?"}
    )
    assert response.status_code == 200
    assert response.json()["result"] == "ham"


def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_special_chars():
    response = client.post(
        "/predict",
        json={"text": "!!! $$$ WIN MONEY NOW $$$ !!!"}
    )
    assert response.status_code == 200
    assert response.json()["result"] in ["spam", "ham"]

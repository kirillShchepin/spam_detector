from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_spam():
    response = client.post("/predict", json={"text": "Win a free prize now!"})
    assert response.status_code == 200
    assert response.json()["result"] in ["spam", "ham"]


def test_predict_ham():
    response = client.post("/predict", json={"text": "Hello, how are you?"})
    assert response.status_code == 200
    assert response.json()["result"] in ["spam", "ham"]

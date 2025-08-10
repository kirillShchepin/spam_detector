from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    response = client.post("/predict", json={"text": "You win a free prize!"})
    assert response.status_code == 200
    assert response.json()["result"] in ["spam", "ham"]
    assert 0 <= response.json()["confidence"] <= 1

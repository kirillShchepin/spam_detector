from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_spam():
    response = client.post("/predict", json={"text": "Win a free iPhone!"})
    assert response.status_code == 200
    json_data = response.json()
    assert "result" in json_data
    assert json_data["result"] in ["spam", "ham"]
    assert 0 <= json_data["confidence"] <= 1


def test_predict_ham():
    response = client.post("/predict", json={"text": "Hello, how are you?"})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["result"] in ["spam", "ham"]
    assert 0 <= json_data["confidence"] <= 1

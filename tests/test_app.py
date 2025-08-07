from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_spam():
    response = client.post("/predict", json={"text": "Win a free iPhone!"})
    assert response.status_code == 200
    assert "result" in response.json()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_spam():
    # Теперь отправляем данные в правильном формате
    response = client.post(
        "/predict",
        json={"text": "Win a free iPhone!"}  # Совпадает с PredictionRequest
    )
    assert response.status_code == 200
    assert "result" in response.json()
    assert response.json()["result"] in ["spam", "ham"]

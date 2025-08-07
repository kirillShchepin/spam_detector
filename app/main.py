from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # Добавляем валидацию данных
from transformers import pipeline
import torch

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str  # Строго типизированный ввод

# Инициализация модели вынесена в отдельную функцию для надежности
def get_model():
    try:
        return pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
            device=-1  # Используем CPU
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

model = get_model()

@app.post("/predict")
async def predict(request: PredictionRequest):  # Используем Pydantic модель
    try:
        result = model(request.text)[0]["label"]
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

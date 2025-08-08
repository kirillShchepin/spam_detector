import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(f"Request: {request.text}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

def get_model():
    try:
        # Указываем return_all_scores=False для получения только лучшего результата
        return pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
            device=-1,
            return_all_scores=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

model = get_model()

# Создаем маппинг меток
LABEL_MAPPING = {"LABEL_0": "ham", "LABEL_1": "spam"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        result = model(request.text)[0]
        # Преобразуем метку в читаемый формат
        return {"result": LABEL_MAPPING.get(result["label"], result["label"])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

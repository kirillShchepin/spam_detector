"""FastAPI приложение для детекции спама с использованием лёгкой модели."""

import logging
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class PredictionRequest(BaseModel):
    """Модель данных для запроса предсказания."""
    text: str


@lru_cache(maxsize=1)
def load_model():
    """Загружает и кэширует модель классификации спама."""
    try:
        model_pipeline = pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
            device="cpu"
        )
        logger.info("Spam detection model loaded successfully")
        return model_pipeline
    except Exception as e:
        logger.error("Model loading failed: %s", str(e), exc_info=True)
        raise


try:
    model = load_model()
except Exception as e:
    logger.critical("Model initialization failed: %s", str(e))
    model = None


@app.get("/")
async def root():
    """Корневой эндпоинт для проверки работы API."""
    return {
        "status": "ok" if model else "error",
        "message": "Spam Detector API",
        "model_loaded": bool(model)
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Эндпоинт для классификации текста на спам или не спам.

    Args:
        request: объект с полем text

    Returns:
        dict: метка классификации и уверенность
    """
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        result = model(request.text)[0]
        return {
            "result": result["label"].lower(),
            "confidence": float(result["score"])
        }
    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction error"
        )

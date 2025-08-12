from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging
from functools import lru_cache


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class PredictionRequest(BaseModel):
    """Модель для входных данных API."""
    text: str


@lru_cache(maxsize=1)
def load_model():
    """Загружает и кэширует модель классификации текста."""
    try:
        model = pipeline(
            "text-classification",
            model="seara/rubert-tiny2-russian-sentiment",
            device="cpu"
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error("Model loading failed: %s", str(e), exc_info=True)
        raise


@app.get("/")
async def root():
    try:
        model = load_model()
    
    except Exception as e:
        logger.critical("Model initialization failed: %s", str(e))
        model = None
    """Корневой эндпоинт для проверки работы API."""
    return {
        "status": "ok" if model else "error",
        "message": "Spam Detector API",
        "model_loaded": bool(model)
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Эндпоинт для классификации текста на спам/не спам.

    Args:
        request: PredictionRequest с полем text

    Returns:
        dict: Результат классификации и уверенность модели

    Raises:
        HTTPException: Если модель не загружена или произошла ошибка
    """
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        result = model(request.text)[0]
        return {
            "result": "spam" if result["label"] == "POSITIVE" else "ham",
            "confidence": float(result["score"]),
            "res": result["label"]
        }
    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Prediction error"
        )

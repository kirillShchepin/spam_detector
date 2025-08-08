from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spam Detector API")


class TextInput(BaseModel):
    """Модель входных данных"""
    text: str


def predict_label(text: str) -> str:
    """Простая проверка на спам по ключевым словам"""
    spam_keywords = ["win", "free", "prize", "cash", "offer"]
    text_lower = text.lower()
    if any(word in text_lower for word in spam_keywords):
        return "spam"
    return "ham"


@app.post("/predict")
async def predict(input_data: TextInput):
    """
    Эндпоинт для получения предсказания
    Принимает JSON: {"text": "..."}
    Возвращает: {"result": "spam" или "ham"}
    """
    result = predict_label(input_data.text)
    return {"result": result}


@app.on_event("shutdown")
async def shutdown_event():
    """Действия при завершении работы приложения"""
    logger.info("Shutting down Spam Detector API")

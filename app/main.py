from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import logging
import os
from transformers import pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spam Detector API")

# Включаем CORS, чтобы можно было вызывать API с любых доменов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Можно указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель Hugging Face
logger.info("Загрузка модели...")
model = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
logger.info("Модель загружена.")

# Маппинг меток модели в привычные значения
LABEL_MAP = {
    "LABEL_0": "ham",   # не спам
    "LABEL_1": "spam"   # спам
}


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Spam Detector API is running"
    }


@app.get("/web")
async def web_interface():
    file_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    return FileResponse(file_path)


class TextInput(BaseModel):
    """Модель входных данных"""
    text: str


def predict_label(text: str) -> str:
    """Предсказание спама с использованием ML-модели"""
    prediction = model(text)
    raw_label = prediction[0]["label"].upper()
    return LABEL_MAP.get(raw_label, raw_label)


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

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

# Включаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для продакшена лучше ограничить
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    """Модель входных данных"""
    text: str


@app.on_event("startup")
async def startup_event():
    """
    Загружаем предобученную модель при старте приложения.
    Модель: fine-tuned BERT для классификации спама
    """
    global spam_detector
    logger.info("Загрузка модели спам-фильтра...")
    spam_detector = pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
        tokenizer="mrm8488/bert-tiny-finetuned-sms-spam-detection"
    )
    logger.info("Модель загружена успешно!")


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


@app.post("/predict")
async def predict(input_data: TextInput):
    """
    Получение предсказания от модели
    """
    prediction = spam_detector(input_data.text)[0]
    label = prediction["label"].lower()
    score = round(prediction["score"], 3)
    return {
        "result": label,
        "confidence": score
    }


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Spam Detector API")

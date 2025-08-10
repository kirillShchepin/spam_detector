from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import pipeline
import logging
import os


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spam Detector API")

# Включаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Можно указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель для классификации спама
model = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)


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
    """Определяем спам или нет с помощью ML-модели"""
    prediction = model(text)[0]["label"]
    if prediction == "label_1":
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

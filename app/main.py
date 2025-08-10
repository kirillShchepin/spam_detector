from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import logging
import os


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Spam Detector API")


# Включаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Spam Detector API is running"
    }


@app.get("/web")
async def web_interface():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))
    return FileResponse(file_path)


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
    """JSON API: {"text": "..."} → {"result": "..."}"""
    result = predict_label(input_data.text)
    return {"result": result}


@app.on_event("shutdown")
async def shutdown_event():
    """Действия при завершении работы приложения"""
    logger.info("Shutting down Spam Detector API")

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем настоящую модель спам-детекции
logger.info("Загружаем ML-модель...")
spam_detector = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
    tokenizer="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)
logger.info("Модель загружена.")

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
    text: str

@app.post("/predict")
async def predict(input_data: TextInput):
    """
    Получение предсказания от настоящей ML-модели.
    """
    result = spam_detector(input_data.text)[0]
    label = result["label"].lower()  # spam или ham
    score = round(result["score"], 4)
    return {"result": label, "confidence": score}

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Spam Detector API")

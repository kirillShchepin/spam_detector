from fastapi import FastAPI
from transformers import pipeline
import torch  # Явный импорт для работы модели

app = FastAPI()

# Загрузка модели спам-детектора
spam_model = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)

@app.post("/predict")
def predict(text: str):
    return {"result": spam_model(text)[0]["label"]}  # "spam" или "ham"

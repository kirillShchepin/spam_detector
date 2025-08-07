from fastapi import FastAPI, Body
from pydantic import BaseModel
from transformers import pipeline
import torch

class TextInput(BaseModel):
    text: str

app = FastAPI()

# Загрузка модели спам-детектора
spam_model = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
)

@app.post("/predict")
def predict(data: TextInput):
    result = spam_model(data.text)[0]
    return {"result": result["label"]}  # "spam" или "ham"

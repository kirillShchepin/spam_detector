from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Используем облегченную модель для избежания ошибок
spam_model = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
    framework="pt"  # Явно указываем PyTorch
)

@app.post("/predict")
def predict(text: str):
    return {"result": spam_model(text)[0]["label"]}

from fastapi import FastAPI
from transformers import pipeline
import torch  # Явный импорт

# Проверка доступности CUDA для отладки
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

app = FastAPI()

try:
    spam_model = pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
        device=-1  # Принудительно используем CPU
    )
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise

@app.post("/predict")
def predict(text: str):
    return {"result": spam_model(text)[0]["label"]}

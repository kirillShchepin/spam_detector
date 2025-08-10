from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import pipeline
import logging
import os
from functools import lru_cache

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

# Загрузка модели с кэшированием
@lru_cache(maxsize=1)
def load_model():
    """Загружаем и кэшируем модель"""
    try:
        model = pipeline(
            "text-classification",
            model="mariagrandury/roberta-base-finetuned-sms-spam-detection",
            device="cpu"  # Для GPU укажите device=0
        )
        
        # Тестовая проверка модели при загрузке
        test_spam = model("Win a free iPhone now! Click here!")[0]
        test_ham = model("Hello, let's meet tomorrow")[0]
        logger.info(f"Model loaded. Test spam: {test_spam}, test ham: {test_ham}")
        
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError("Failed to load model")

try:
    model = load_model()
except Exception as e:
    logger.critical(f"Critical error: {e}")
    model = None

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {
        "status": "ok" if model else "error",
        "message": "Spam Detector API is running",
        "model_loaded": bool(model)
    }

@app.post("/predict")
async def predict(input_data: TextInput):
    """
    Эндпоинт для проверки текста на спам.
    Принимает JSON: {"text": "ваш текст"}
    Возвращает: {"result": "spam" или "ham"}
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Логируем входные данные
        logger.info(f"Predict request: {input_data.text[:50]}...")
        
        # Получаем предсказание
        prediction = model(input_data.text)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Определяем результат (метки специфичны для этой модели)
        label = prediction["label"]
        confidence = prediction["score"]
        
        if label == "LABEL_1":
            result = "spam"
        else:
            result = "ham"
        
        # Логируем детали
        logger.info(f"Result: {result} (confidence: {confidence:.2f})")
        
        return {
            "result": result,
            "confidence": float(confidence),
            "raw_prediction": prediction
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/web")
async def web_interface():
    """Веб-интерфейс для тестирования"""
    file_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    return FileResponse(file_path)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API")

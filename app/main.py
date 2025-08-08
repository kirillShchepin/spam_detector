from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging
from logging.handlers import RotatingFileHandler
import os


# Настройка логирования
os.makedirs("logs", exist_ok=True)  # Создаем папку для логов

logger = logging.getLogger("spam_detector")
logger.setLevel(logging.INFO)

# Обработчик для файла (макс. 5 МБ, 3 резервные копии)
file_handler = RotatingFileHandler(
    "logs/spam_detector.log",
    maxBytes=5*1024*1024,
    backupCount=3,
    encoding="utf-8"
)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Обработчик для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = FastAPI(title="Spam Detector API")


class PredictionRequest(BaseModel):
    text: str


# Маппинг меток модели
LABEL_MAPPING = {
    "LABEL_0": "ham",
    "LABEL_1": "spam"
}


def load_model():
    """Загрузка и настройка модели"""
    try:
        logger.info("Loading model...")
        model = pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection",
            device=-1,  # Используем CPU
            return_all_scores=False
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.critical(f"Model loading failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Model initialization error"
        )


model = load_model()


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Классифицирует текст как спам или не спам
    
    Параметры:
    - text: строка для анализа
    
    Возвращает:
    - {"result": "spam"|"ham", "confidence": float}
    """
    try:
        logger.info(f"Processing request: {request.text[:100]}...")  # Логируем первые 100 символов
        
        # Получаем предсказание
        prediction = model(request.text)[0]
        logger.debug(f"Raw prediction: {prediction}")
        
        # Преобразуем метку
        label = LABEL_MAPPING.get(prediction["label"], prediction["label"])
        confidence = float(prediction["score"])
        
        logger.info(f"Prediction: {label} (confidence: {confidence:.2f})")
        
        return {
            "result": label,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Действия при запуске приложения"""
    logger.info("Starting Spam Detector API")


@app.on_event("shutdown")
async def shutdown_event():
    """Действия при завершении работы"""
    logger.info("Shutting down Spam Detector API")

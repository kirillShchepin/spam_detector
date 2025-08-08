from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging
from logging.handlers import RotatingFileHandler
import os
from contextlib import asynccontextmanager

# Настройка логирования
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("spam_detector")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    "logs/spam_detector.log",
    maxBytes=5*1024*1024,
    backupCount=3,
    encoding="utf-8"
)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events"""
    logger.info("Starting Spam Detector API")
    yield
    logger.info("Shutting down Spam Detector API")


app = FastAPI(title="Spam Detector API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    text: str


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
            device=-1,
            top_k=1
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
    """Классифицирует текст как спам или не спам"""
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    try:
        logger.info(f"Processing request: {request.text[:100]}...")
        prediction = model(request.text)[0]
        logger.debug(f"Raw prediction: {prediction}")

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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging
from functools import lru_cache


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the ML model."""
    try:
        # Using smaller model for testing
        model = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device="cpu"
        )
        logger.info("Model test prediction: %s", model("Free prize!")[0])
        return model
    except Exception as e:
        logger.error("Loading failed: %s", str(e), exc_info=True)
        raise


try:
    model = load_model()
except Exception as e:
    logger.critical("INIT FAILED: %s", str(e))
    model = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if model else "error",
        "model": "loaded" if model else "failed"
    }


@app.post("/predict")
async def predict(input_data: BaseModel):
    """Make spam/ham prediction for input text.

    Args:
        input_data: Pydantic model with 'text' field

    Returns:
        dict: Prediction result with confidence score

    Raises:
        HTTPException: If model isn't loaded or prediction fails
    """
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable"
        )

    try:
        result = model(input_data.text)[0]
        return {
            "result": "spam" if result["label"] == "LABEL_1" else "ham",
            "confidence": result["score"]
        }
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

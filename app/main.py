from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

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
    try:
        result = spam_model(data.text)[0]
        return {
            "result": result["label"],
            "score": result["score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

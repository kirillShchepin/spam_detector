from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spam Detector API")

# Поддержка CORS (можно ограничить список доменов в allow_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модель для JSON-запросов
class TextInput(BaseModel):
    text: str


# Корневой маршрут
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Spam Detector API is running"
    }


# Простейшая модель классификации
def predict_label(text: str) -> str:
    spam_keywords = ["win", "free", "prize", "cash", "offer"]
    text_lower = text.lower()
    if any(word in text_lower for word in spam_keywords):
        return "spam"
    return "ham"


# HTML-форма
@app.get("/web", response_class=HTMLResponse)
async def web_form():
    return """
    <html>
        <head>
            <title>Spam Detector</title>
        </head>
        <body>
            <h2>Spam Detector</h2>
            <form action="/predict" method="post">
                <input type="text" name="text" placeholder="Введите текст">
                <button type="submit">Проверить</button>
            </form>
        </body>
    </html>
    """


# POST-запрос через JSON
@app.post("/predict")
async def predict_json(input_data: TextInput):
    result = predict_label(input_data.text)
    return {"result": result}


# POST-запрос через HTML-форму
@app.post("/predict", response_class=HTMLResponse)
async def predict_form(text: str = Form(...)):
    result = predict_label(text)
    return f"<h3>Результат: {result}</h3>"


# Событие при завершении работы
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Spam Detector API")

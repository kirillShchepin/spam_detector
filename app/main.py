from fastapi import FastAPI
from pydantic import BaseModel
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spam Detector API")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Spam Detector API is running"
    }

class TextInput(BaseModel):
    """Модель входных данных"""
    text: str

def predict_label(text: str) -> str:
    """Простая проверка на спам по ключевым словам"""
    spam_keywords = ["win", "free", "prize", "cash", "offer"]
    text_lower = text.lower()
    if any(word in text_lower for word in spam_keywords):
        return "spam"
    return "ham"

@app.post("/predict")
async def predict(input_data: TextInput):
    """JSON API: {"text": "..."} → {"result": "..."}"""
    result = predict_label(input_data.text)
    return {"result": result}

# Пытаемся подключить поддержку HTML-форм
try:
    from fastapi import Form
    from fastapi.responses import HTMLResponse

    @app.get("/web", response_class=HTMLResponse)
    async def form_page():
        return """
        <html>
            <body>
                <h2>Spam Detector</h2>
                <form action="/web" method="post">
                    <input type="text" name="text" placeholder="Введите текст">
                    <input type="submit" value="Проверить">
                </form>
            </body>
        </html>
        """

    @app.post("/web", response_class=HTMLResponse)
    async def predict_form(text: str = Form(...)):
        result = predict_label(text)
        return f"""
        <html>
            <body>
                <h2>Результат: {result}</h2>
                <a href="/web">Назад</a>
            </body>
        </html>
        """

except ImportError:
    logger.warning("python-multipart не установлен — эндпоинт /web отключён")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Spam Detector API")

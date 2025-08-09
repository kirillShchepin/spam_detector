FROM python:3.10-slim

# Устанавливаем зависимости
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Запускаем с портом из переменной окружения, которую передает Yandex.Cloud
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

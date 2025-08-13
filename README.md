# spam_detector
Web-приложение для классификации спама в SMS
Основные возможности:
ML-модель: Использует предобученный rubert-tiny2 для анализа текста.
REST API: Эндпоинты / (проверка работы) и /predict (классификация).
CI/CD: Автоматические тесты (GitHub Actions) и деплой в Yandex Cloud.


Структура проекта
spam_detector/
│
├── .github/workflows/          # GitHub Actions для CI/CD
│   ├── ci.yml                 # Тесты при push/pull-request
│   └── deploy.yml             # Деплой в Yandex Cloud (Docker)
│
├── app/                       # Основное приложение (FastAPI)
│   ├── __init__.py            # Пустой файл для инициализации модуля
│   └── main.py                # API для классификации спама:
│
├── tests/                     # Тесты
│   ├── __init__.py            # Пустой файл
│   └── test_app.py            # Тесты API (корень и /predict)
│
├── .flake8                    # Конфиг линтера (max-line-length=88)
├── Dockerfile                 # Сборка Docker-образа (Python 3.10 + FastAPI)
├── index.html                 # Веб-интерфейс для тестирования API
└── requirements.txt           # Зависимости:
                               # - fastapi, uvicorn (API)
                               # - transformers, torch (ML-модель)
                               # - pytest, flake8 (тесты и линтинг)

Как сделать запрос и проверить работу?
1. https://bbaohv58j553juds883e.containers.yandexcloud.net/docs - переходим по ссылке
2. Разворачиваем POST /predict
3. Нажимаем кнопку "Try it out"
4. В поле "text": "string" вводим запрос. Вместо string. Пример запроса: "text": "hi, my friend"
5. Нажимаем кнопку "Execute" и получаем ответ

Примеры запроса:

1. 
запрос:
{
  "text": "hi, my friend"
}
ответ:
{
  "result": "ham",
  "confidence": 0.4268045723438263,
  "res": "positive"
}
То есть данное письмо не является спамом

2. 
запрос:
{
  "text": "URGENT! You've won a FREE $1,000,000 Walmart gift card! Click NOW: bit.ly/fake-link123 Claim before 24 hours expire!!!"
}
ответ:
{
  "result": "spam",
  "confidence": 0.7815194725990295,
  "res": "negative"
}
То есть данной письмо с большой вероятностью является спамом







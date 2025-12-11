# Тестовый скрипт для отправки запроса агенту через локальный FastAPI-эндпоинт
# Выполняет POST /ask и выводит финальный ответ модели

import requests
import json

url = "http://localhost:8000/ask"
payload = {
    "question": "Сколько времени потребуется гепарду, чтобы преодолеть Большой Каменный мост?"
}
# Сколько времени потребуется гепарду, чтобы преодолеть Большой Каменный мост?
# Какая площадь у Красной Площади?

resp = requests.post(url, json=payload)
resp.raise_for_status()

data = resp.json()

print(data["answer"]) 
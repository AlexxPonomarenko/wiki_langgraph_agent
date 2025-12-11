from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Инициализация клиента OpenAI. Ключ должен быть в переменной окружения OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_llm_system_prompt(system_prompt: str, user_content: str, *,
                          model: str = "gpt-4o-mini",
                          temperature: float = 0.2) -> str:
    """
    Выполняет единичный запрос к LLM с заданными system prompt и текстом пользователя

    Параметры:
        system_prompt: Системная инструкция, задающая стиль и поведение модели
        user_content: Основной текст запроса от пользователя
        model: Название модели, используемой для вызова
        temperature: Степень стохастичности генерации

    Возвращает:
        Строку — текст ответа ассистента
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    content = resp.choices[0].message.content
    return content.strip() if content else ""

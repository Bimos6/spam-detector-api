import os

from dotenv import load_dotenv

load_dotenv()

_client = None
if os.getenv("OPENAI_API_KEY"):
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


PROMPT_TEMPLATE = """Ты — система фильтрации спама. Пользователь получил текст,
который модель классифицировала как СПАМ.

Текст: {text}

Объясни в одном-двух предложениях по-русски, какие признаки указывают
на то, что это спам. Будь краток и конкретен."""


def explain_prediction(text: str) -> str:
    """Возвращает объяснение от LLM. Если ключа нет — fallback."""
    if _client is None:
        return (
            "LLM недоступен (нет API-ключа). "
            "Классификатор определил спам по TF-IDF-признакам."
        )

    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
            ],
            max_tokens=120,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM недоступен из-за ошибки: {type(e).__name__}"
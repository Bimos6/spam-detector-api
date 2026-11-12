# Spam Detector API

Классификатор SMS/email на спам с REST API. Полный ML-пайплайн:
данные → PostgreSQL → TF-IDF → LogisticRegression → FastAPI → Docker.

## Стек

- **ML:** Python 3.11, Scikit-learn, Pandas, NumPy
- **БД:** PostgreSQL (в Docker), SQLAlchemy + psycopg2
- **API:** FastAPI, Pydantic, Uvicorn
- **LLM:** OpenAI API (опционально, graceful fallback без ключа)
- **Деплой:** Docker, Docker Compose

## Результаты

| Метрика | Значение |
|---|---|
| **Accuracy** | см. `models/metrics.json` |
| **Precision** | см. `models/metrics.json` |
| **Recall** | см. `models/metrics.json` |
| **ROC-AUC** | см. `models/metrics.json` |
| **CV ROC-AUC (5-fold)** | см. `models/metrics.json` |

![ROC Curve](models/plots/roc_curve.png)
![Confusion Matrix](models/plots/confusion_matrix.png)

## Быстрый старт (Docker Compose)

```bash
# 1. Скопировать переменные окружения
cp .env.example .env

# 2. (Опционально) вставь OPENAI_API_KEY в .env

# 3. Скачать датасет
# SMS Spam Collection: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Положить в data/spam.csv

# 4. Поднять БД
docker compose up -d db

# 5. Обучить модель
docker compose run --rm api python -m src.train

# 6. Запустить API
docker compose up -d api
```

API: http://localhost:8000/docs

## Ручной запуск (для разработки)

```bash
# 1. Виртуальное окружение
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# 2. Зависимости
pip install -r requirements.txt

# 3. Postgres в Docker
docker compose up -d db

# 4. Обучение
python -m src.train

# 5. API
uvicorn src.main:app --reload --port 8000
```

## Эндпоинты

### `GET /health`
Health-check.

### `GET /stats`
Статистика классов в БД:
```json
{"0": 4825, "1": 747}
```

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won $1000. Call now!"}'
```

Ответ:
```json
{
  "text": "Congratulations! You won $1000. Call now!",
  "is_spam": 1,
  "confidence": 0.9873,
  "label": "spam",
  "llm_explanation": "Текст содержит обещание крупного выигрыша и призыв к немедленному действию — классические признаки мошенничества."
}
```

## Архитектура

```
CSV ──► PostgreSQL ──► SQL-запрос ──► Pandas DataFrame
                                           │
                                           ▼
                               train_test_split (stratify)
                                           │
                            ┌──────────────┴──────────────┐
                            ▼                             ▼
                       Cross-Validation              Fit на train
                       (проверка переобучения)            │
                                                          ▼
                                             Pipeline(TF-IDF + LogReg)
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼               ▼               ▼
                                      Metrics         Plots         model.pkl
                                                                        │
                                                                        ▼
                                                                    FastAPI
                                                                        │
                                                                        ▼
                                                                    Docker
```

## Что покрыто

- ✅ Python, чистый код, venv
- ✅ Pandas, NumPy, Scikit-learn
- ✅ Метрики: Accuracy, Precision, Recall, ROC-AUC
- ✅ Кросс-валидация (борьба с переобучением)
- ✅ SQL + PostgreSQL (SQLAlchemy + psycopg2)
- ✅ LLM (OpenAI API + graceful fallback)
- ✅ Обучение и развёртывание (FastAPI + Docker)
- ✅ Docker Compose для инфраструктуры

## Переменные окружения

| Переменная | Описание | По умолчанию |
|---|---|---|
| `POSTGRES_USER` | Пользователь БД | `spam_user` |
| `POSTGRES_PASSWORD` | Пароль | `spam_password` |
| `POSTGRES_DB` | Имя БД | `spam_db` |
| `POSTGRES_HOST` | Хост | `localhost` / `db` в Docker |
| `POSTGRES_PORT` | Порт | `5432` |
| `OPENAI_API_KEY` | Ключ OpenAI | пусто (fallback) |
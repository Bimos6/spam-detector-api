from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.database import get_stats
from src.llm_helper import explain_prediction

MODEL_PATH = Path("models/model.pkl")

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Модель не найдена: {MODEL_PATH}. "
        f"Сначала запусти: python -m src.train"
    )
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Spam Detector API",
    description="Классификатор SMS/email на спам. TF-IDF + LogisticRegression.",
    version="1.0.0",
)


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["Congratulations! You won $1000. Call now!"],
    )


class PredictionResponse(BaseModel):
    text: str
    is_spam: int
    confidence: float
    label: str
    llm_explanation: str


@app.get("/health")
def health() -> dict:
    """Health-check для Docker/Kubernetes."""
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    """Статистика по классам из БД (SQL GROUP BY)."""
    return get_stats()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest) -> PredictionResponse:
    """Классифицирует текст и объясняет решение."""
    try:
        proba = float(model.predict_proba([request.text])[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    is_spam = int(proba >= 0.5)
    label = "spam" if is_spam else "ham"

    explanation = (
        explain_prediction(request.text)
        if is_spam
        else "Не спам — объяснение не требуется."
    )

    return PredictionResponse(
        text=request.text,
        is_spam=is_spam,
        confidence=round(proba, 4),
        label=label,
        llm_explanation=explanation,
    )
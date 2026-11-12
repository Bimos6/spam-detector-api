import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.database import init_db, load_from_db

MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("models/metrics.json")
PLOTS_DIR = Path("models/plots")


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def evaluate(y_true, y_pred, y_proba, name: str) -> dict:
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def plot_confusion(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()


def plot_roc(y_true, y_proba, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    init_db()
    df = load_from_db(min_length=5)

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[split] train={len(X_train)}  test={len(X_test)}")

    model = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"[cv]   ROC-AUC на 5-фолдах: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_proba, "LogReg+TFIDF")
    metrics["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_roc_auc_std"] = round(float(cv_scores.std()), 4)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))

    print("\n[metrics]")
    print(json.dumps(metrics, indent=2))
    print("\n[classification report]")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    plot_confusion(y_test, y_pred, PLOTS_DIR / "confusion_matrix.png")
    plot_roc(y_test, y_proba, PLOTS_DIR / "roc_curve.png")
    print(f"\n[plots] Сохранены в {PLOTS_DIR}/")

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[save] Модель: {MODEL_PATH}")
    print(f"[save] Метрики: {METRICS_PATH}")


if __name__ == "__main__":
    main()
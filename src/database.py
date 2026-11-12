import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()

# --- Конфиг БД из окружения ---
DB_USER = os.getenv("POSTGRES_USER", "spam_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "spam_password")
DB_NAME = os.getenv("POSTGRES_DB", "spam_db")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2))
def wait_for_db() -> None:
    """Ждём, пока Postgres поднимется. Полезно в Docker."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("[db] Postgres готов к работе")


def init_db(csv_path: str = "data/spam.csv") -> None:
    """Создаёт таблицу и заливает в неё данные из CSV."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV не найден: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    df = df[["label", "text"]].dropna()
    df["label"] = (df["label"] == "spam").astype(int)
    df["text"] = df["text"].astype(str).str.strip()

    wait_for_db()

    # Запись через pandas.to_sql
    df.to_sql("messages", engine, if_exists="replace", index=False)
    print(f"[db] Загружено {len(df)} записей в таблицу 'messages'")

    # Проверяем, что данные на месте
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM messages")).scalar()
    print(f"[db] В таблице messages теперь {total} строк")


def load_from_db(min_length: int = 5) -> pd.DataFrame:
    """
    Выгружает данные через SQL-запрос.
    Фильтр по длине + проверка на NULL — чтобы показать работу с SQL.
    """
    query = text("""
        SELECT label, text
        FROM messages
        WHERE text IS NOT NULL
          AND LENGTH(text) >= :min_len
    """)

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"min_len": min_length})

    print(f"[db] Выгружено {len(df)} записей (SQL)")
    return df


def get_stats() -> dict:
    """Доп. функция — статистика по классам через SQL (пригодится в API)."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT label, COUNT(*) AS cnt
            FROM messages
            GROUP BY label
            ORDER BY label
        """)).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


if __name__ == "__main__":
    init_db()
    df = load_from_db()
    print(df.head())
    print("Статистика:", get_stats())
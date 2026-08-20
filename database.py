"""App database (SQLite) — users table."""

import os
import time
import sqlite3
import threading

APP_DB = os.path.join(os.path.dirname(__file__), "app.db")

_db_lock = threading.Lock()

_NEW_USER_COLUMNS = {
    "notify_email": "TEXT",
    "notify_by_email": "INTEGER NOT NULL DEFAULT 0",
    "notify_by_telegram": "INTEGER NOT NULL DEFAULT 0",
    "telegram_chat_id": "TEXT",
}

def _init_db():
    conn = sqlite3.connect(APP_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at REAL
        )
    """)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    for name, coldef in _NEW_USER_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE users ADD COLUMN {name} {coldef}")
    conn.commit()
    conn.close()

_init_db()

def get_user_by_email(email: str):
    conn = sqlite3.connect(APP_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id: int):
    conn = sqlite3.connect(APP_DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def create_user(email: str, hashed_password: str, role: str = "agent"):
    with _db_lock:
        conn = sqlite3.connect(APP_DB)
        cur = conn.execute(
            "INSERT INTO users (email, hashed_password, role, created_at) VALUES (?, ?, ?, ?)",
            (email, hashed_password, role, time.time()),
        )
        conn.commit()
        user_id = cur.lastrowid
        conn.close()
    return get_user_by_email(email) or {"id": user_id, "email": email, "role": role}

def list_users():
    conn = sqlite3.connect(APP_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, email, role, created_at FROM users ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def list_notification_targets():
    conn = sqlite3.connect(APP_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, email, notify_email, notify_by_email, notify_by_telegram, telegram_chat_id "
        "FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_user(user_id: int) -> bool:
    with _db_lock:
        conn = sqlite3.connect(APP_DB)
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
    return deleted

def update_user_settings(user_id: int, notify_email, notify_by_email: bool, notify_by_telegram: bool):
    with _db_lock:
        conn = sqlite3.connect(APP_DB)
        conn.execute(
            "UPDATE users SET notify_email = ?, notify_by_email = ?, notify_by_telegram = ? WHERE id = ?",
            (notify_email, int(notify_by_email), int(notify_by_telegram), user_id),
        )
        conn.commit()
        conn.close()
    return get_user_by_id(user_id)

def update_user_password(user_id: int, hashed_password: str):
    with _db_lock:
        conn = sqlite3.connect(APP_DB)
        conn.execute("UPDATE users SET hashed_password = ? WHERE id = ?", (hashed_password, user_id))
        conn.commit()
        conn.close()

def update_telegram_chat_id(user_id: int, chat_id: str):
    with _db_lock:
        conn = sqlite3.connect(APP_DB)
        conn.execute("UPDATE users SET telegram_chat_id = ? WHERE id = ?", (chat_id, user_id))
        conn.commit()
        conn.close()

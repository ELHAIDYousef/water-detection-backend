"""Event log (SQLite) — overflow event history."""

import time
import sqlite3
import threading

from fastapi import APIRouter, Depends

from config import EVENTS_DB
from auth import get_current_user

_events_lock = threading.Lock()

def _init_events_db():
    conn = sqlite3.connect(EVENTS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            camera_id TEXT,
            camera_name TEXT,
            place TEXT,
            reference TEXT,
            coverage_percent REAL
        )
    """)
    conn.commit()
    conn.close()

_init_events_db()

def log_event(cam, coverage):
    try:
        with _events_lock:
            conn = sqlite3.connect(EVENTS_DB)
            conn.execute(
                "INSERT INTO events (timestamp, camera_id, camera_name, place, reference, coverage_percent) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), cam["id"], cam["name"], cam["place"], cam["reference"], round(coverage, 2)),
            )
            conn.commit()
            conn.close()
    except Exception as e:
        print("[events] log error:", e)

router = APIRouter(dependencies=[Depends(get_current_user)])

@router.get("/events")
def list_events(q: str = "", limit: int = 100):
    conn = sqlite3.connect(EVENTS_DB)
    conn.row_factory = sqlite3.Row
    if q:
        like = f"%{q}%"
        rows = conn.execute(
            "SELECT * FROM events WHERE camera_name LIKE ? OR place LIKE ? OR reference LIKE ? "
            "ORDER BY id DESC LIMIT ?", (like, like, like, limit),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"events": [dict(r) for r in rows], "count": len(rows)}

# ---- analytics aggregation (SQL-side, no raw-row post-processing) ----

def count_events_today() -> int:
    conn = sqlite3.connect(EVENTS_DB)
    row = conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE date(timestamp, 'unixepoch', 'localtime') = date('now', 'localtime')"
    ).fetchone()
    conn.close()
    return row[0] if row else 0

def count_events_since_days(days: int) -> int:
    conn = sqlite3.connect(EVENTS_DB)
    row = conn.execute(
        "SELECT COUNT(*) FROM events "
        "WHERE date(timestamp, 'unixepoch', 'localtime') >= date('now', ?, 'localtime')",
        (f"-{days - 1} days",),
    ).fetchone()
    conn.close()
    return row[0] if row else 0

def events_by_day(days: int = 7):
    conn = sqlite3.connect(EVENTS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT date(timestamp, 'unixepoch', 'localtime') AS date, COUNT(*) AS count "
        "FROM events "
        "WHERE date(timestamp, 'unixepoch', 'localtime') >= date('now', ?, 'localtime') "
        "GROUP BY date ORDER BY date ASC",
        (f"-{days - 1} days",),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def events_by_place():
    conn = sqlite3.connect(EVENTS_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT place, COUNT(*) AS count FROM events GROUP BY place ORDER BY count DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def count_total_events() -> int:
    conn = sqlite3.connect(EVENTS_DB)
    row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
    conn.close()
    return row[0] if row else 0

@router.get("/analytics/summary")
def analytics_summary():
    return {
        "events_today": count_events_today(),
        "events_last_7_days": count_events_since_days(7),
        "events_by_day": events_by_day(7),
        "events_by_place": events_by_place(),
        "total_events": count_total_events(),
    }

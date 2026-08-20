"""Telegram bot and email (Gmail SMTP) integration — sending alerts and account pairing."""

import os
import random
import smtplib
import threading
import time
from datetime import datetime
from email.message import EmailMessage

import requests
from fastapi import APIRouter, Depends, HTTPException
from dotenv import load_dotenv

from database import get_user_by_id, update_telegram_chat_id, list_notification_targets
from auth import get_current_user

OVERFLOW_NOTIFY_COOLDOWN_SEC = 5 * 60

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_BOT_USERNAME = os.environ.get("TELEGRAM_BOT_USERNAME", "")
TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

SMTP_HOST = os.environ.get("SMTP_HOST") or "smtp.gmail.com"
SMTP_PORT = int(os.environ.get("SMTP_PORT") or 587)
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM_NAME = os.environ.get("SMTP_FROM_NAME") or "OCP Laverie Alerts"

def send_telegram(chat_id, text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        print("[telegram] TELEGRAM_BOT_TOKEN not set - cannot send message")
        return False
    try:
        resp = requests.post(
            f"{TELEGRAM_API_BASE}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        if not resp.ok:
            print(f"[telegram] sendMessage failed ({resp.status_code}): {resp.text}")
            return False
        data = resp.json()
        if not data.get("ok"):
            print(f"[telegram] sendMessage rejected: {data}")
            return False
        return True
    except Exception as e:
        print("[telegram] sendMessage error:", e)
        return False

def send_email(to_email: str, subject: str, html_body: str) -> bool:
    if not SMTP_USER or not SMTP_PASSWORD:
        print("[email] SMTP_USER/SMTP_PASSWORD not set - cannot send email")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
        msg["To"] = to_email
        msg.set_content("This email requires an HTML-capable client to view.")
        msg.add_alternative(html_body, subtype="html")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print("[email] send error:", e)
        return False

# ---- overflow alerts: per-camera cooldown to avoid flooding ----
_overflow_cooldown_lock = threading.Lock()
_last_overflow_notified_at = {}

def notify_overflow(cam, coverage):
    cam_id = cam["id"]
    now = time.time()
    with _overflow_cooldown_lock:
        last = _last_overflow_notified_at.get(cam_id, 0)
        if now - last < OVERFLOW_NOTIFY_COOLDOWN_SEC:
            return
        _last_overflow_notified_at[cam_id] = now

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"Water overflow alert - {cam['name']}"
    html_body = (
        "<h2>Water overflow detected</h2>"
        "<p>"
        f"<b>Camera:</b> {cam['name']}<br>"
        f"<b>Place:</b> {cam['place']}<br>"
        f"<b>Reference:</b> {cam['reference']}<br>"
        f"<b>Coverage:</b> {coverage:.2f}%<br>"
        f"<b>Time:</b> {timestamp}"
        "</p>"
    )
    text_body = (
        "Water overflow alert\n"
        f"Camera: {cam['name']}\n"
        f"Place: {cam['place']}\n"
        f"Reference: {cam['reference']}\n"
        f"Coverage: {coverage:.2f}%\n"
        f"Time: {timestamp}"
    )

    for user in list_notification_targets():
        if user.get("notify_by_email") and user.get("notify_email"):
            try:
                send_email(user["notify_email"], subject, html_body)
            except Exception as e:
                print(f"[notify] email failed for {user.get('email')}:", e)
        if user.get("notify_by_telegram") and user.get("telegram_chat_id"):
            try:
                send_telegram(user["telegram_chat_id"], text_body)
            except Exception as e:
                print(f"[notify] telegram failed for {user.get('email')}:", e)

_last_update_id = 0

def _get_updates():
    global _last_update_id
    if not TELEGRAM_BOT_TOKEN:
        print("[telegram] TELEGRAM_BOT_TOKEN not set - cannot poll updates")
        return []
    try:
        resp = requests.get(
            f"{TELEGRAM_API_BASE}/getUpdates",
            params={"offset": _last_update_id + 1, "timeout": 0},
            timeout=10,
        )
        if not resp.ok:
            print(f"[telegram] getUpdates failed ({resp.status_code}): {resp.text}")
            return []
        data = resp.json()
        if not data.get("ok"):
            print(f"[telegram] getUpdates rejected: {data}")
            return []
        results = data.get("result", [])
        for update in results:
            _last_update_id = max(_last_update_id, update.get("update_id", 0))
        return results
    except Exception as e:
        print("[telegram] getUpdates error:", e)
        return []

# ---- pairing codes: code -> user_id ----
_pairing_lock = threading.Lock()
_PAIRING_CODES = {}

def _generate_pairing_code(user_id: int) -> str:
    with _pairing_lock:
        for _ in range(20):
            code = f"OCP-{random.randint(1000, 9999)}"
            if code not in _PAIRING_CODES:
                _PAIRING_CODES[code] = user_id
                return code
        code = f"OCP-{random.randint(1000, 9999)}-{user_id}"
        _PAIRING_CODES[code] = user_id
        return code

def _consume_pending_updates():
    """Scan recent Telegram messages for a pending pairing code and link the sender's chat."""
    updates = _get_updates()
    if not updates:
        return
    with _pairing_lock:
        pending = dict(_PAIRING_CODES)
    for update in updates:
        message = update.get("message") or {}
        text = (message.get("text") or "").strip().upper()
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None or not text:
            continue
        for code, user_id in pending.items():
            if code in text:
                update_telegram_chat_id(user_id, str(chat_id))
                with _pairing_lock:
                    _PAIRING_CODES.pop(code, None)
                pending.pop(code, None)
                break

router = APIRouter(prefix="/me/telegram", tags=["telegram"])

@router.post("/connect")
def connect_telegram(user: dict = Depends(get_current_user)):
    code = _generate_pairing_code(user["id"])
    bot_link = f"https://t.me/{TELEGRAM_BOT_USERNAME}?start={code}" if TELEGRAM_BOT_USERNAME else None
    return {"code": code, "bot_link": bot_link}

@router.get("/status")
def telegram_status(user: dict = Depends(get_current_user)):
    _consume_pending_updates()
    current = get_user_by_id(user["id"])
    return {"linked": bool(current and current.get("telegram_chat_id"))}

@router.post("/test")
def test_telegram(user: dict = Depends(get_current_user)):
    current = get_user_by_id(user["id"])
    chat_id = current.get("telegram_chat_id") if current else None
    if not chat_id:
        raise HTTPException(status_code=400, detail="Telegram is not linked for this account")
    ok = send_telegram(chat_id, "OCP water detection: this is a test notification.")
    if not ok:
        raise HTTPException(status_code=502, detail="Failed to send Telegram message")
    return {"status": "ok"}

email_router = APIRouter(prefix="/me/email", tags=["email"])

@email_router.post("/test")
def test_email(user: dict = Depends(get_current_user)):
    current = get_user_by_id(user["id"])
    notify_email = current.get("notify_email") if current else None
    if not notify_email:
        raise HTTPException(status_code=400, detail="No notify_email is set for this account")
    ok = send_email(
        notify_email,
        "OCP Laverie Alerts - Test Email",
        "<p>This is a test notification from the OCP water detection backend.</p>",
    )
    if not ok:
        raise HTTPException(status_code=502, detail="Failed to send test email")
    return {"status": "ok"}

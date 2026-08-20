"""
Water Detection Backend — complete standalone FastAPI app (CPU or GPU).

Features:
  - POST /detect/image           single-image detection (sync)
  - POST /detect/video           video detection (async job)
  - GET  /jobs/{job_id}          poll a video job
  - GET  /cameras                45 simulated cameras + live status (round-robin worker)
  - GET  /cameras/{id}/frame     latest annotated frame for one camera (JPEG)
  - GET  /cameras/{id}/video     raw source video for one camera (no overlay)
  - GET  /events                 overflow event log (SQLite, newest first)

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
Cameras read videos from a ./videos folder next to this file.
"""

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import device
from database import list_users, create_user
from auth import hash_password
import auth
import users
import me
import notifications
import detection
import events
import cameras

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(me.router)
app.include_router(notifications.router)
app.include_router(notifications.email_router)
app.include_router(detection.router)
app.include_router(events.router)
app.include_router(cameras.router)

def _seed_admin():
    if list_users():
        return
    admin_email = os.environ.get("ADMIN_EMAIL") or "admin@ocp.local"
    admin_password = os.environ.get("ADMIN_PASSWORD") or "admin123"
    create_user(admin_email, hash_password(admin_password), role="admin")
    print(f"[auth] no users found - created default admin '{admin_email}'")

_seed_admin()
cameras.start_camera_worker()

@app.get("/")
def health():
    return {"status": "ok", "message": "water detection backend running", "device": device}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

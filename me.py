"""Self-service account settings and notification preferences."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from database import update_user_settings, update_user_password
from auth import get_current_user, hash_password, verify_password

router = APIRouter(prefix="/me", tags=["me"])

class SettingsResponse(BaseModel):
    email: str
    role: str
    notify_email: Optional[str] = None
    notify_by_email: bool
    notify_by_telegram: bool
    telegram_linked: bool

def _to_settings_response(user: dict) -> SettingsResponse:
    return SettingsResponse(
        email=user["email"],
        role=user["role"],
        notify_email=user.get("notify_email"),
        notify_by_email=bool(user.get("notify_by_email")),
        notify_by_telegram=bool(user.get("notify_by_telegram")),
        telegram_linked=bool(user.get("telegram_chat_id")),
    )

@router.get("/settings", response_model=SettingsResponse)
def get_settings(user: dict = Depends(get_current_user)):
    return _to_settings_response(user)

class UpdateSettingsRequest(BaseModel):
    notify_email: Optional[str] = None
    notify_by_email: bool = False
    notify_by_telegram: bool = False

@router.put("/settings", response_model=SettingsResponse)
def update_settings(payload: UpdateSettingsRequest, user: dict = Depends(get_current_user)):
    updated = update_user_settings(
        user["id"], payload.notify_email, payload.notify_by_email, payload.notify_by_telegram
    )
    return _to_settings_response(updated)

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

@router.post("/password")
def change_password(payload: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    if not verify_password(payload.old_password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    update_user_password(user["id"], hash_password(payload.new_password))
    return {"status": "ok"}

"""Admin-only user management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from database import list_users, create_user, get_user_by_email, delete_user
from auth import hash_password, require_admin

router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_admin)])

class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str = "agent"

@router.get("")
def get_users():
    return {"users": list_users()}

@router.post("")
def add_user(payload: CreateUserRequest):
    if payload.role not in ("admin", "agent"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'agent'")
    if get_user_by_email(payload.email):
        raise HTTPException(status_code=409, detail="A user with this email already exists")
    user = create_user(payload.email, hash_password(payload.password), role=payload.role)
    return {"id": user["id"], "email": user["email"], "role": user["role"], "created_at": user["created_at"]}

@router.delete("/{user_id}")
def remove_user(user_id: int, current_user: dict = Depends(require_admin)):
    if current_user["id"] == user_id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    if not delete_user(user_id):
        raise HTTPException(status_code=404, detail="No such user")
    return {"status": "ok", "deleted": user_id}

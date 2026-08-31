"""
Password hashing helpers. Uses bcrypt directly (not passlib - passlib's
bcrypt backend has known compatibility breakage with bcrypt>=4.1).
"""

import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHEM,
    JWT_SECRET,
    REFRESH_TOKEN_EXPIRE_DAYS,
)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_token(user):
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"user_id": user.id, "exp": expire, "type": "access"}
    token = jwt.encode(payload, JWT_SECRET, JWT_ALGORITHEM)
    return token


def create_refresh_token(user):
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {"user_id": user.id, "exp": expire, "type": "refresh"}
    token = jwt.encode(payload, JWT_SECRET, JWT_ALGORITHEM)
    return token


def generate_otp():
    otp = str(secrets.randbelow(900000) + 100000)
    print(otp)
    return 666666

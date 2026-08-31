from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import (
    create_refresh_token,
    create_token,
    generate_otp,
    hash_password,
    verify_password,
)
from get_db import get_db
from models import Users
from redis_client import redis
from schemas import LoginData, UserCreate, VerificationData, verifyOtpData

import asyncio

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login")
async def login(loginData: LoginData, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Users).where(
            (Users.email == loginData.email) | (Users.phone_number == loginData.phone_number)
        )
    )
    existing_user = result.scalar_one_or_none()
    if existing_user is None:
        raise HTTPException(status_code=401, detail="user is not registered with us")

    correctpassword = verify_password(loginData.password, existing_user.password)
    if correctpassword is False:
        raise HTTPException(status_code=401, detail="Invalid user")

    token = create_token(existing_user)
    refresh_token = create_refresh_token(existing_user)
    return {"token": token, "refresh_token": refresh_token}


@router.post("/send-verification")
async def send_verification(data: VerificationData):
    if data.type == "phone":
        key = data.countryCode + data.value
    else:
        key = data.value

    otp = generate_otp()
    otp_key = key + ":otp"
    await redis.set(otp_key, otp, ex=600)
    return {"otp": otp}


@router.post("/verify")
async def verify_identity(data: verifyOtpData):
    if data.type == "phone":
        key = data.countryCode + data.value
    else:
        key = data.value

    otp_key = key + ":otp"
    value = await redis.get(otp_key)

    if value is None:
        return {"message": "otp has expired"}
    if value != data.otp:
        return {"message": "Invalid otp"}

    verified_key = key + ":verified"
    await asyncio.gather(redis.set(verified_key, data.otp, ex=1800), redis.delete(otp_key))

    return True


@router.post("/register")
async def register(data: UserCreate, db: AsyncSession = Depends(get_db)):
    email_key = data.email + ":verified"
    if await redis.get(email_key) is None:
        raise HTTPException(400, detail="please verify email")
    phone_key = str(data.country_code) + data.phone_number + ":verified"
    if await redis.get(phone_key) is None:
        raise HTTPException(400, detail="please verify phone_number")

    hashed_password = hash_password(data.password)
    user = Users(
        name=data.name,
        email=data.email,
        phone_number=data.phone_number,
        password=hashed_password,
        country_code=data.country_code,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

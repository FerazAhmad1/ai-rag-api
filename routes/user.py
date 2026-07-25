from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth import hash_password
from get_db import get_db
from models import Users
from schemas import UserCreate, UserOut

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(Users).where(
            (Users.email == user.email) | (Users.phone_number == user.phone_number)
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email or phone number already exists.",
        )

    new_user = Users(
        name=user.name,
        email=user.email,
        phone_number=user.phone_number,
        country_code=user.country_code,
        password=hash_password(user.password),
    )
    db.add(new_user)

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email or phone number already exists.",
        )

    return UserOut(
        id=new_user.id,
        name=new_user.name,
        email=new_user.email,
        phone_number=new_user.phone_number,
        country_code=new_user.country_code,
    )

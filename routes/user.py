from fastapi import APIRouter,Depends
from sqlalchemy.ext.asyncio import AsyncSession
from get_db import get_db
from schemas import UserCreate
router = APIRouter(prefix="/users",tags=["Users"])

@router.post("/")
async def create_user(
    user:UserCreate,
    db:AsyncSession = Depends(get_db)
):
    print(user)
    return {"message":"User received"}
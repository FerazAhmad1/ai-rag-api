import asyncio

from sqlalchemy import text
from database import engine

async def test_connection():
    async with engine.connect() as conn:
        db = await conn.execute(text("select current_database();"))
        user = await conn.execute(text("SELECT current_user;"))
        print(f"Database : {db.scalar()}")
        print (f"user : {user.scalar()}")
asyncio.run(test_connection())



import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

if ENVIRONMENT == "production":
    user = os.getenv("PROD_DB_USER")
    password = os.getenv("PROD_DB_PASSWORD")
    host = os.getenv("PROD_DB_HOST")
    port = os.getenv("PROD_DB_PORT")
    database = os.getenv("PROD_DB_NAME")
else:
    user = os.getenv("LOCAL_DB_USER")
    password = os.getenv("LOCAL_DB_PASSWORD")
    host = os.getenv("LOCAL_DB_HOST")
    port = os.getenv("LOCAL_DB_PORT")
    database = os.getenv("LOCAL_DB_NAME")

password = quote_plus(password or "")

DATABASE_URL = (
    f"postgresql+asyncpg://"
    f"{user}:{password}@{host}:{port}/{database}"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass
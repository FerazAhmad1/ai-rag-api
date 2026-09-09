from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import engine
from embeddings import create_jina_client
from generation import create_deepseek_client
from routes.auth import router as auth_router
from routes.user import router as user_router
from routes.document import router as document_router
from routes.search import router as search_router
from routes.ask import router as ask_router
from routes.generate import router as generate_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.jina_client = create_jina_client()
    app.state.deepseek_client = create_deepseek_client()
    try:
        yield
    finally:
        await app.state.jina_client.aclose()
        await app.state.deepseek_client.aclose()
        await engine.dispose()


app = FastAPI(lifespan=lifespan)

# Wide open for local dev - tighten to specific origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def first_api():
    return {"message":"first api created"}

app.include_router(auth_router)
app.include_router(user_router)
app.include_router(document_router)
app.include_router(search_router)
app.include_router(ask_router)
app.include_router(generate_router)
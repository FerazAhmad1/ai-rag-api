from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.user import router as user_router
from routes.document import router as document_router
from routes.search import router as search_router
app = FastAPI()

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

app.include_router(user_router)
app.include_router(document_router)
app.include_router(search_router)
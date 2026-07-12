from fastapi import FastAPI
from routes.user import router as user_router
from routes.document import router as document_router
app = FastAPI()

@app.get("/")
async def first_api():
    return {"message":"first api created"}

app.include_router(user_router)
app.include_router(document_router)
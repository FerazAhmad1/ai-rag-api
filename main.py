from fastapi import FastAPI
from routes.user import router as user_router
app = FastAPI()

@app.get("/")
async def first_api():
    return {"message":"first api created"}

app.include_router(user_router)
from pydantic import BaseModel
class UserCreate(BaseModel):
    name: str
    email: str
    phone_number: str
    country_code: int
    password: str


from pydantic import BaseModel
class UserCreate(BaseModel):
    name: str
    email: str
    phone_number: str
    country_code: int
    password: str


class PageTextOut(BaseModel):
    page_number: int
    text: str
    char_count: int


class DocumentExtractionOut(BaseModel):
    filename: str
    page_count: int
    pages: list[PageTextOut]


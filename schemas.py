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


class ChunkOut(BaseModel):
    chunk_index: int
    page_number: int
    text: str
    token_count: int
    char_count: int


class DocumentChunkingOut(BaseModel):
    filename: str
    page_count: int
    total_chunks: int
    chunks: list[ChunkOut]


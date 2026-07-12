from datetime import datetime

from pydantic import BaseModel, Field
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


class DocumentOut(BaseModel):
    id: int
    filename: str
    page_count: int
    chunk_count: int
    uploaded_at: datetime


class SearchRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResultOut(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    chunk_index: int
    content: str
    similarity_score: float


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResultOut]


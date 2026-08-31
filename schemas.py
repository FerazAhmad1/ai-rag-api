from datetime import datetime
from typing import Optional,Literal
from pydantic import BaseModel, Field,model_validator
class UserCreate(BaseModel):
    name: str
    email: str
    phone_number: str
    country_code: int
    # max_length=72 approximates bcrypt's 72-byte limit (exact for ASCII;
    # multi-byte UTF-8 passwords could still exceed 72 bytes under 72 chars).
    password: str = Field(min_length=8, max_length=72)
    confirm_password: str = Field(min_length=8, max_length=72)

    @model_validator(mode="after")
    def validate_passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("passwords do not match")
        return self


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    phone_number: str
    country_code: int


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

class LoginData(BaseModel):
    email:Optional[str]
    phone_number:Optional[str]
    password:str

    @model_validator(mode="after")
    def validate_login(self):
        if not self.email and not self.phone_number:
            raise ValueError ("Either email or phone must be provided")
        return self


class VerificationData(BaseModel):
    type:Literal["email","phone"]
    value:str
    countryCode:Optional[str] = None

    @model_validator(mode="after")
    def validate_country_code(self):
        if self.type == "phone" and not self.countryCode:
            raise ValueError("countryCode is required when type is phone")
        return self

class verifyOtpData(VerificationData):
    otp:str = Field(min_length=6,max_length=6)








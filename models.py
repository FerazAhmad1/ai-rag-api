from datetime import datetime, timezone

from sqlalchemy import Column,Integer,String,Boolean,ForeignKey,Text,DateTime
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from database import Base

class Users(Base):
    __tablename__="users"
    id=Column(Integer,primary_key=True,index=True)
    name=Column(String(100),unique=True,index=True)
    email=Column(String(255),unique=True,index=True,nullable=False)
    phone_number = Column(String(15),unique=True,index=True,nullable=False)
    country_code = Column(Integer,nullable=False,index=True)
    password = Column(String)


# Must match the output dimension of the sentence-transformers model used for
# embedding generation (all-MiniLM-L6-v2 -> 384). Change this and re-migrate
# if the embedding model ever changes.
EMBEDDING_DIM = 384


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    page_count = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="chunks")
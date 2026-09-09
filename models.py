from datetime import datetime, timezone

from sqlalchemy import Column,Integer,String,Boolean,ForeignKey,Text,DateTime
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from database import Base

class Users(Base):
    __tablename__="users"
    id=Column(Integer,primary_key=True,index=True)
    name=Column(String(100),index=True)
    email=Column(String(255),unique=True,index=True,nullable=False)
    phone_number = Column(String(15),unique=True,index=True,nullable=False)
    country_code = Column(Integer,nullable=False,index=True)
    password = Column(String)


# Must match embeddings.EMBEDDING_DIMENSIONS (Jina jina-embeddings-v5-text-
# small, requested at 512 dims via the API's `dimensions` param - one of a
# fixed discrete set Jina supports: 32/64/128/256/512/1024, not an arbitrary
# value). Change both and re-migrate if the embedding model/dimension ever
# changes.
EMBEDDING_DIM = 512


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
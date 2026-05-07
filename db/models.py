from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, DateTime, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    highlighter_ranking = Column(JSON, nullable=True) 
    pen_ranking = Column(JSON, nullable=True)         

    documents = relationship("Document", back_populates="owner")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    title = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    page_number = Column(Integer)
    original_text = Column(Text, nullable=False)
    meta_data = Column(JSON, nullable=True) 

    document = relationship("Document", back_populates="chunks")
    importance = relationship("ImportanceResult", back_populates="chunk", uselist=False)

class ImportanceResult(Base):
    __tablename__ = "importance_results"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id", ondelete="CASCADE"), unique=True)
    score = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    keywords = Column(JSON, nullable=True)

    chunk = relationship("DocumentChunk", back_populates="importance")
    questions = relationship("Question", back_populates="importance_data")

class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    importance_id = Column(Integer, ForeignKey("importance_results.id", ondelete="CASCADE"))
    
    question_type = Column(String)
    difficulty = Column(String)
    question_text = Column(Text, nullable=False)
    options = Column(JSON, nullable=True)
    answer = Column(Text, nullable=False)
    explanation = Column(Text, nullable=False)
    
    importance_data = relationship("ImportanceResult", back_populates="questions")
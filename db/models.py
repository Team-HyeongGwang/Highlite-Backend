from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, ForeignKey, DateTime, Text, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from db.database import Base
from sqlalchemy.dialects.postgresql import UUID
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, nullable=False)
    provider = Column(String, default="local")
    profile_image_url = Column(String, nullable=True)
    join_date = Column(DateTime(timezone=True), server_default=func.now())
    
    highlighter_ranking = Column(JSON, nullable=True) 
    pen_ranking = Column(JSON, nullable=True)         

    documents = relationship("Document", back_populates="owner")

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    group_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    title = Column(String, nullable=False)
    doc_type = Column(String, server_default='combined')
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    page_number = Column(Integer)
    original_text = Column(Text, nullable=False)
    meta_data = Column(JSON, nullable=True) 
    embedding = Column(Vector(1536), nullable=True)

    __table_args__ = (
        Index(
            'ix_chunks_embedding_hnsw',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )

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
    
    # ← 추가: 한 번에 생성된 문제 묶음 ID
    quiz_group_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    # ← 추가: 문제 생성 시간
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    question_type = Column(String)
    difficulty = Column(String)
    question_text = Column(Text, nullable=False)
    options = Column(JSON, nullable=True)
    answer = Column(Text, nullable=False)
    explanation = Column(Text, nullable=False)
    
    importance_data = relationship("ImportanceResult", back_populates="questions")

class QuizResult(Base):
    __tablename__ = "quiz_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    
    # 어떤 문제 묶음에 대한 채점인지 연결
    quiz_group_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    # 시험 점수 및 통계
    total_questions = Column(Integer, nullable=False, default=0)
    correct_count = Column(Integer, nullable=False, default=0)
    score_percent = Column(Integer, nullable=False, default=0) 
    # 시도 단계 (첫 시도, 재시도 등)
    attempt_phase = Column(String, default="first_attempt") 
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # 답변과 채점 결과를 저장하는 관계
    answers = relationship("UserAnswer", back_populates="quiz_result", cascade="all, delete-orphan")

class UserAnswer(Base):
    __tablename__ = "user_answers"

    id = Column(Integer, primary_key=True, index=True)
    quiz_result_id = Column(Integer, ForeignKey("quiz_results.id", ondelete="CASCADE"))
    question_id = Column(Integer, ForeignKey("questions.id", ondelete="CASCADE"))
    
    # 사용자 응답 및 채점 결과
    user_answer = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 관계 설정
    quiz_result = relationship("QuizResult", back_populates="answers")
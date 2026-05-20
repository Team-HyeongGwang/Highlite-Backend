from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, ForeignKey, DateTime, Text, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
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
    group_id = Column(String, index=True, nullable=False)  
    title = Column(String, nullable=False)
    doc_type = Column(String, server_default='combined')
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
    embedding = Column(Vector(1536), nullable=True)  # OpenAI text-embedding-3-small

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
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    
    # 💡 시험 점수 및 통계
    total_questions = Column(Integer, nullable=False, default=0)
    correct_count = Column(Integer, nullable=False, default=0)
    score_percent = Column(Integer, nullable=False, default=0) 
    
    # 💡 1회차(원본) 풀이인지, N회차(재생성) 풀이인지 추적
    attempt_phase = Column(String, default="first_attempt") 
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 관계 설정 (Relationships)
    answers = relationship("UserAnswer", back_populates="quiz_result", cascade="all, delete-orphan")
    # owner = relationship("User", back_populates="quiz_results")
    # document = relationship("Document", back_populates="quiz_results")


class UserAnswer(Base):
    __tablename__ = "user_answers"

    id = Column(Integer, primary_key=True, index=True)
    quiz_result_id = Column(Integer, ForeignKey("quiz_results.id", ondelete="CASCADE"))
    question_id = Column(Integer, ForeignKey("questions.id", ondelete="CASCADE"))
    
    # 💡 유저의 응답 데이터
    user_answer = Column(Text, nullable=True) # 사용자가 제출한 답 (빈칸/미입력 고려하여 True)
    is_correct = Column(Boolean, nullable=False, default=False) # AI 재생성을 위한 핵심 O/X 지표
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 관계 설정 (Relationships)
    quiz_result = relationship("QuizResult", back_populates="answers")
    # question = relationship("Question", back_populates="user_answers")
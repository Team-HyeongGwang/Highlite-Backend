from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from uuid import UUID

class QuestionGenerateRequest(BaseModel):
    group_id: str = Field(..., description="문제를 출제할 문서 세트 ID")
    question_count: int = Field(10, ge=10, le=50, description="생성할 문제 개수 (10~50개)")

class QuestionItem(BaseModel):
    chunk_id: int
    question_id: int = Field(..., description="DB questions 테이블 id")
    question_type: str = Field(..., description="'multiple_choice', 'ox', 'fill_in_the_blank'")
    question_text: str
    options: Optional[Dict[str, str]] = Field(None, description="객관식일 경우 4지선다 보기")
    answer: str
    explanation: str
    question_number: int = Field(..., description="문제 번호 (1, 2, 3...)")
    priority: int = Field(..., description="중요도 순위 (1/2/3)")
    source_type: str = Field(..., description="'highlight' 또는 'pen'")
    page_number: int = Field(..., description="출처 페이지 번호")

class QuestionGenerateResponse(BaseModel):
    questions: List[QuestionItem]

class RegenerateRequest(BaseModel):
    question_id: int
    importance_id: int
    context_text: str
    keywords: List[str]
    question_type: str
    feedback_type: str
    retry_count: int = 0

class RegenerateResponse(BaseModel):
    question_id: int
    question_type: str
    question_text: str
    options: Optional[Dict[str, str]] = None
    answer: str
    explanation: str

class SubmitAnswerRequest(BaseModel):
    user_id: int
    document_id: UUID
    attempt_phase: str = "first_attempt"
    answers: List[dict] = Field(..., description="[{question_id: 1, submitted_answer: '②'}, ...]")

class AnswerResult(BaseModel):
    question_id: int
    submitted_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str

class SubmitAnswerResponse(BaseModel):
    total: int
    correct: int
    wrong: int
    results: List[AnswerResult]

class RegenerateFromWrongRequest(BaseModel):
    user_id: int
    document_id: UUID
    group_id: str
    question_count: int = Field(10, ge=10, le=50)

# ────────────────────────────────────────
# /question/list 용 schemas
# ────────────────────────────────────────
class AttemptItem(BaseModel):
    quiz_result_id: int
    round: int
    created_at: str
    q_num: int
    score: Optional[int] = None
    attempt_phase: str

class DocumentItem(BaseModel):
    document_id: UUID
    title: str
    upload_date: str
    total_count: int
    attempts: List[AttemptItem]

class QuestionListRequest(BaseModel):
    user_id: int
    document_id: Optional[UUID] = None

class QuestionListResponse(BaseModel):
    documents: List[DocumentItem]
    
class DeleteQuizResultRequest(BaseModel):
    user_id: int
    quiz_result_ids: List[int]

class DeleteQuizResultResponse(BaseModel):
    deleted_count: int
    message: str
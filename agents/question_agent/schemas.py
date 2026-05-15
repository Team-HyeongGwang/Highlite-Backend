# agents/question_agent/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class QuestionGenerateRequest(BaseModel):
    group_id: str = Field(..., description="문제를 출제할 문서 세트 ID")
    question_count: int = Field(10, ge=10, le=30, description="생성할 문제 개수 (10~30개)")

class QuestionItem(BaseModel):
    chunk_id: int
    question_type: str = Field(..., description="'multiple_choice', 'ox', 'fill_in_the_blank'")
    question_text: str
    options: Optional[Dict[str, str]] = Field(None, description="객관식일 경우 4지선다 보기")
    answer: str
    explanation: str

class QuestionGenerateResponse(BaseModel):
    questions: List[QuestionItem]

# 피드백 재생성 (router.py에서 이동)
class RegenerateRequest(BaseModel):
    question_id: int
    importance_id: int
    context_text: str
    keywords: List[str]
    question_type: str
    feedback_type: str  # "ambiguous" / "wrong_answer" / "unclear_explanation" / "irrelevant"
    retry_count: int = 0

class RegenerateResponse(BaseModel):
    question_id: int
    question_type: str
    question_text: str
    options: Optional[Dict[str, str]] = None
    answer: str
    explanation: str
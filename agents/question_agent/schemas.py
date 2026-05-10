from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from common.schemas import VisualCue

class QuestionGenerateRequest(BaseModel):
    group_id: str = Field(..., description="문제를 출제할 문서 세트 ID")
    question_count: int = Field(10, ge=10, le=30, description="생성할 문제 개수 (명세서 기준 10~30개)")

class QuestionItem(BaseModel):
    """생성된 개별 문제 정보"""
    chunk_id: int # 어떤 조각에서 나왔는지 추적용
    question_type: str = Field(..., description="'multiple_choice', 'ox', 'fill_in_the_blank'")
    question_text: str
    options: Optional[Dict[str, str]] = Field(None, description="객관식일 경우 4지선다 보기")
    answer: str
    explanation: str

class QuestionGenerateResponse(BaseModel):
    questions: List[QuestionItem]
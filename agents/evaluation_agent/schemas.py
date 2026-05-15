from pydantic import BaseModel, Field
from typing import Dict, Optional

class QuestionReviewRequest(BaseModel):
    group_id: str
    source_chunk_text: str = Field(..., description="팩트 체크를 위한 원본 텍스트")
    
    # 평가할 문제 데이터
    question_text: str
    options: Optional[Dict[str, str]]
    answer: str
    explanation: str

# 검토 후 내리는 판정 결과
class QuestionReviewResponse(BaseModel):
    is_approved: bool = Field(..., description="최종 출제 승인 여부 (True면 DB 저장, False면 다시 생성)")
    quality_score: int = Field(..., description="문제 퀄리티 점수 (1~10점)")
    feedback: str = Field(..., description="반려 사유 (예: '원본 텍스트에 없는 내용이 포함됨', '정답이 2개임')")
    
    # "이렇게 고치면 어때?" 하고 바로 수정안을 던져줄 수도 있음
    suggested_revision_text: Optional[str] = Field(None, description="제안하는 문제 지문 수정안")
    suggested_revision_options: Optional[Dict[str, str]] = None


class FeedbackType(str, Enum):
    wrong_answer     = "wrong_answer"       # 정답이 틀림
    ambiguous        = "ambiguous"          # 문제가 애매함
    not_in_source    = "not_in_source"      # 원본에 없는 내용
    multiple_correct = "multiple_correct"   # 복수 정답 존재

class FeedbackRequest(BaseModel):
    question_id: int
    feedback_type: FeedbackType

class FeedbackResponse(BaseModel):
    regenerated: bool
    new_question_text: Optional[str] = None
    new_options: Optional[Dict[str, str]] = None
    new_answer: Optional[str] = None
    new_explanation: Optional[str] = None
    message: str
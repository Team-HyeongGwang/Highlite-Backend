from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class QuestionGenerateRequest(BaseModel):
    importance_id: int
    context_text: str = Field(..., description="문제의 근거가 되는 원문")
    keywords: List[str]
    difficulty: str = "중"
    question_type: str = "multiple_choice" 

class QuestionGenerateResponse(BaseModel):
    question_text: str
    options: Optional[Dict[str, str]] = None
    answer: str
    explanation: str = Field(..., description="정답 해설")
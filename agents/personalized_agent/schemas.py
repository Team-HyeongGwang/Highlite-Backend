from pydantic import BaseModel, Field
from typing import List

# 프론트엔드에서 학생이 답을 제출할 때 보낼 양식
class AnswerSubmissionRequest(BaseModel):
    user_id: int
    group_id: str
    question_id: int = Field(..., description="학생이 푼 문제의 DB ID")
    user_answer: str = Field(..., description="학생이 제출한 답")
    is_correct: bool = Field(..., description="단순 정답 여부 (프론트에서 비교 후 전송하거나 채점 API 거친 후의 결과)")

class PersonalizationResponse(BaseModel):
    weakness_concepts: List[str] = Field(..., description="사용자가 반복해서 틀리는 개념")
    weakness_source_file: str = Field(..., description="취약 개념이 포함된 원본 PDF 파일명") 
    personalized_advice: str = Field(..., description="AI가 제안하는 맞춤형 학습 방향")
    next_recommendation: List[str] = Field(..., description="다음에 풀어볼 만한 키워드 추천")
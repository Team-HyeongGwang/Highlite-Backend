from pydantic import BaseModel, Field
from uuid import UUID

# 프론트엔드에서 학생이 답을 제출할 때 보낼 양식
class AnswerSubmissionRequest(BaseModel):
    user_id: int
    group_id: UUID
    question_id: int = Field(..., description="학생이 푼 문제의 DB ID")
    user_answer: str = Field(..., description="학생이 제출한 답")
    is_correct: bool = Field(..., description="단순 정답 여부 (프론트에서 비교 후 전송하거나 채점 API 거친 후의 결과)")
    attempt_phase: str = Field("first_attempt", description="first_attempt 또는 re_attempt")

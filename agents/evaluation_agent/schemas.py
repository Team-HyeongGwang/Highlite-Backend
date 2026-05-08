from pydantic import BaseModel, Field

class EvaluationRequest(BaseModel):
    user_id: int
    question_id: int
    user_answer: str

class EvaluationResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: str
    feedback: str = Field(..., description="정답/오답 시 간단한 피드백")
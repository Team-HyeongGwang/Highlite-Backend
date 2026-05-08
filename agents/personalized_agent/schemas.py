from pydantic import BaseModel, Field
from typing import List

class PersonalizationRequest(BaseModel):
    user_id: int

class PersonalizationResponse(BaseModel):
    weakness_concepts: List[str] = Field(..., description="사용자가 반복해서 틀리는 개념")
    personalized_advice: str = Field(..., description="AI가 제안하는 맞춤형 학습 방향")
    next_recommendation: List[str] = Field(..., description="다음에 풀어볼 만한 키워드 추천")
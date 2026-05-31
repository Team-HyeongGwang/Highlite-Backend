from pydantic import BaseModel, Field
from uuid import UUID
from typing import Optional

# ── 제출 스키마 ──────────────────────────────────────────
class BatchAnswerItem(BaseModel):
    question_id: int
    user_answer: str
    is_correct: bool

class BatchSubmissionRequest(BaseModel):
    user_id: int
    group_id: UUID
    attempt_phase: str = Field("first_attempt", description="first_attempt 또는 re_attempt")
    answers: list[BatchAnswerItem]

# ── 조회 응답 스키마 ─────────────────────────────────────
class AttemptSummary(BaseModel):
    id: int
    round: int
    date: str
    total: int
    correct: int
    wrong: int

class DocumentQuizResults(BaseModel):
    id: str
    title: str
    total_count: int
    attempts: list[AttemptSummary]

class WrongAnswerItem(BaseModel):
    id: str
    imp: str
    type: str
    text: str
    options: Optional[list[str]]
    my_ans: str
    correct: str
    source: str
    exp: str

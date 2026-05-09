from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from common.schemas import VisualCue

class ImportanceRequest(BaseModel):
    group_id: str
    chunk_id: int
    doc_type: str 
    original_text: str
    meta_data: List[VisualCue] = Field(default_factory=list)
    highlighter_ranking: Dict[str, int] = Field(..., description="형광펜 중요도 순위")
    pen_ranking: Dict[str, int] = Field(..., description="펜 중요도 순위")

class ImportanceResponse(BaseModel):
    score: float = Field(..., description="최종 중요도 점수 (0~10)")
    reasoning: str = Field(..., description="점수 부여 근거")
    keywords: List[str] = Field(..., description="핵심 키워드")
    summary: Optional[str] = Field(None, description="조각 요약")
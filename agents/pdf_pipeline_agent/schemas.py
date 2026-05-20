# schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
import re


from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from uuid import UUID, uuid4

class PDFChunk(BaseModel):
    """RAG agent에 넘기는 단위 청크 (계층 구조 제거)"""
    id: UUID = Field(default_factory=uuid4)
    db_id: Optional[int] = None              # DB 저장 후 할당될 ID
    pdf_name: str
    page: int
    paragraph_index: int
    content: str                             # 모든 텍스트(본문/형광펜/손글씨)는 무조건 여기
    embedding: Optional[List[float]] = None  # pgvector 저장용
    
    # ── 🎨 시각적 속성(메타데이터) ──
    handwriting_color: Optional[str] = None  # 값이 있으면 "아, content가 손글씨구나!"
    highlight_color: Optional[str] = None    # 값이 있으면 "아, content가 형광펜이구나!"

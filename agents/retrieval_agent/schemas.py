from pydantic import BaseModel, Field
from typing import List, Optional

class ChunkExtraction(BaseModel):
    page_number: int
    original_text: str
    meta_data: List[dict] = Field(default_factory=list)

class DocumentCreateRequest(BaseModel):
    user_id: int
    title: str
    doc_type: str = Field("textbook", description="textbook 또는 summary_note")

class RetrievalResponse(BaseModel):
    document_id: int
    chunks_count: int
    message: str = "문서 분석 및 벡터 저장 완료"
from pydantic import BaseModel, Field
from typing import List
from common.schemas import VisualCue

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


# PDF에서 추출된 청크 데이터
class ChunkExtraction(BaseModel):
    page_number: int
    original_text: str
    meta_data: List[VisualCue] = Field(default_factory=list) 

# 파일 1개에 대한 기본 정보
class DocumentMetadata(BaseModel):
    title: str = Field(..., description="파일 이름")
    doc_type: str = Field("textbook", description="'textbook' 또는 'summary_note'")

# 여러 개의 파일을 한 번에 업로드할 때의 요청 
class BatchUploadRequest(BaseModel):
    user_id: int
    group_id: str = Field(..., description="같이 업로드된 파일들을 하나로 묶는 세트 ID (예: 'set-123')")
    documents_info: List[DocumentMetadata] = Field(..., description="업로드할 파일 정보 리스트")

# 다중 문서 처리 완료 응답
class BatchRetrievalResponse(BaseModel):
    group_id: str = Field(..., description="처리 완료된 세트 ID")
    processed_documents_count: int = Field(..., description="성공적으로 처리된 문서 개수")
    total_chunks_count: int = Field(..., description="DB에 저장된 총 청크 개수")
    message: str = "다중 문서 분석 및 벡터 DB 저장 완료"

class RAGPipelineRequest(BaseModel):
    document_id: int
    chunks: List[PDFChunk]
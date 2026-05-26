from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from common.schemas import VisualCue

from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID, uuid4


class PDFChunk(BaseModel):
    """RAG agent에 넘기는 단위 청크 (계층 구조 제거)"""
    id: UUID = Field(default_factory=uuid4)
    db_id: Optional[int] = None              # DB 저장 후 할당될 ID
    page: int
    paragraph_index: int
    content: str                             # 모든 텍스트(본문/형광펜/손글씨)는 무조건 여기
    embedding: Optional[List[float]] = None  # pgvector 저장용

    # ── 🎨 시각적 속성(메타데이터) ──
    handwriting_color: Optional[str] = None
    highlight_color: Optional[str] = None
    is_underline: bool = False               # 밑줄 여부
    is_circled: bool = False                 # 동그라미/박스 여부
    is_image: bool = False                   # 이미지/도표 여부 (content에 설명문 저장)

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

class ExtractedChunk(BaseModel):
    content: str = Field(description="문단 단위의 텍스트 내용 또는 이미지 설명. (필기가 부가 설명이면 여기에 자연스럽게 포함)")
    handwriting_color: Optional[str] = Field(default=None, description="손필기 색상 (예: red, blue). 없으면 null")
    highlight_color: Optional[str] = Field(default=None, description="형광펜 색상 (예: yellow). 없으면 null")
    is_underline: bool = Field(default=False, description="밑줄 여부")
    is_circled: bool = Field(default=False, description="동그라미/박스 표시 여부")
    is_image: bool = Field(default=False, description="이 청크가 이미지/도표인지 여부")

class PageOutput(BaseModel):
    chunks: List[ExtractedChunk] = Field(description="페이지에서 추출된 청크 목록")
# schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4

# 형광펜·손필기·주석 메타데이터
class HighlightAnnotation(BaseModel):
    has_handwriting: bool = False
    has_highlight: bool = False
    has_annotation: bool = False
    handwriting_content: Optional[str] = None   # [손필기: ...] 추출 내용
    highlight_content: Optional[str] = None     # [형광펜: ...] 추출 내용
    annotation_content: Optional[str] = None    # [주석: ...] 추출 내용


# extract_from_pdf 결과를 RAG agent에 넘기는 단위 청크
class PDFChunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    pdf_name: str                               # 원본 PDF 파일명
    page: int                                   # 페이지 번호
    paragraph_index: int                        # 페이지 내 문단 순서
    content: str                                # 이미지에서 추출된 텍스트
    annotation: HighlightAnnotation             # 손필기·형광펜 메타데이터
    embedding: Optional[list[float]] = None     # pgvector에 저장될 벡터
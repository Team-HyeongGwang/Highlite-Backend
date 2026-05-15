# schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
import re


class HighlightAnnotation(BaseModel):
    """형광펜·손필기 메타데이터"""
    has_handwriting: bool = False
    has_highlight: bool = False
    handwriting_content: Optional[str] = None
    highlight_content: Optional[str] = None


class PDFChunk(BaseModel):
    """RAG agent에 넘기는 단위 청크"""
    id: UUID = Field(default_factory=uuid4)
    pdf_name: str
    page: int
    paragraph_index: int
    content: str                              # 순수 텍스트 ([태그] 제거된)
    annotation: HighlightAnnotation
    embedding: Optional[list[float]] = None  # pgvector 저장용


def parse_chunk(raw: dict, pdf_name: str) -> PDFChunk:
    """
    extract_from_pdf 결과 dict → PDFChunk 변환.
    content 안의 [손필기: ...], [형광펜: ...] 태그를 파싱합니다.
    """
    content = raw["content"]

    def extract_tag(tag: str) -> Optional[str]:
        match = re.search(rf"\[{tag}: (.+?)\]", content)
        return match.group(1) if match else None

    handwriting = extract_tag("손필기")
    highlight = extract_tag("형광펜")

    clean_content = re.sub(r"\[(손필기|형광펜): .+?\]", "", content).strip()

    return PDFChunk(
        pdf_name=pdf_name,
        page=raw["page"],
        paragraph_index=raw["paragraph_index"],
        content=clean_content,
        annotation=HighlightAnnotation(
            has_handwriting=handwriting is not None,
            has_highlight=highlight is not None,
            handwriting_content=handwriting,
            highlight_content=highlight,
        ),
    )
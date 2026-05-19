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
    handwriting_color: Optional[str] = None
    highlight_color: Optional[str] = None


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
    content = raw["content"]

    def extract_tag(tag: str) -> tuple[Optional[str], Optional[str]]:
        # [손필기: 내용 | 색상: 파란색] 형태 파싱
        match = re.search(rf"\[{tag}: (.+?)(?:\s*\|\s*색상:\s*(.+?))?\]", content)
        if match:
            return match.group(1).strip(), match.group(2).strip() if match.group(2) else None
        return None, None

    handwriting, handwriting_color = extract_tag("손필기")
    highlight, highlight_color = extract_tag("형광펜")

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
            handwriting_color=handwriting_color,
            highlight_color=highlight_color,
        ),
    )
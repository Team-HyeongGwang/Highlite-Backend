import os
import base64
import fitz  # PyMuPDF
from typing import TypedDict
from pathlib import Path
from db.models import Document

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from agents.pdf_pipeline_agent.schemas import PDFChunk, parse_chunk 
from agents.retrieval_agent.service import rag_chain 
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

# Gemini로 바꿀 수도 있음
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# pdf -> image로 바꾸어서 내용 인식
def extract_from_pdf(pdf_path: Path) -> list[dict]:

    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num, page in enumerate(doc, start=1):
        # 페이지 → 이미지 (bytes)
        pix = page.get_pixmap(dpi=200)
        b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")

        response = llm.invoke([
            SystemMessage(content=(
                    "당신은 문서 텍스트 추출 전문가입니다. "
                    "이미지에서 본문 텍스트를 그대로 추출하고, 문단은 빈 줄(\\n\\n)로 구분해 주세요. "
                    "손필기로 쓰여진 텍스트는 반드시 [손필기: 내용 | 색상: 색깔] 형태로 표시하세요. 예시: [손필기: 중요한 내용 | 색상: 파란색] "
                    "형광펜으로 강조된 텍스트는 반드시 [형광펜: 내용 | 색상: 색깔] 형태로 표시하세요. 예시: [형광펜: 강조된 내용 | 색상: 노란색] "
                    "슬라이드 상단/하단의 강의명, 교수명, 학교명, 페이지 번호, 날짜 등 메타 정보는 추출하지 마세요. "
                    "텍스트 외 다른 말은 절대 하지 마세요. 추출할 수 없다는 말도 하지 마세요."
            )),
            
            HumanMessage(content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {"type": "text", "text": f"페이지 {page_num} 텍스트를 추출해 주세요."},
            ]),
        ])

        page_text = response.content.strip()
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]

        for para_idx, content in enumerate(paragraphs, start=1):
            pages.append({
                "page": page_num,
                "paragraph_index": para_idx,
                "content": content,
            })

    doc.close()
    return pages

# Document 저장
async def create_document(
    pdf_path: Path,
    user_id: int,
    group_id: str,
    session: AsyncSession,
) -> Document:
    document = Document(
        user_id=user_id,
        group_id=group_id,
        title=pdf_path.stem,
        doc_type="combined", # 수정 필요
    )
    session.add(document)
    await session.flush() 
    return document

# PDFChunk 생성
async def create_pdf_chunks(
    pdf_path: Path,
    user_id: int,
    group_id: str,
    session: AsyncSession,
) -> None:
    try:
        document = await create_document(pdf_path, user_id, group_id, session)
        print(f"[1] document 생성 완료 id={document.id}")
        
        pdf_name = pdf_path.stem
        raw_chunks = extract_from_pdf(pdf_path)
        print(f"[2] PDF 파싱 완료 count={len(raw_chunks)}")
        
        chunks = [parse_chunk(raw, pdf_name) for raw in raw_chunks]
        print(f"[3] 청크 변환 완료 count={len(chunks)}")
        
        await send_chunks_to_retrieval_agent(chunks, document.id, session)
        print(f"[4] 임베딩 + 저장 완료")
        
    except Exception as e:
        print(f"에러 발생: {e}")
        raise

# retrieval_agent에 결과 전달
async def send_chunks_to_retrieval_agent(chunks: list[PDFChunk], document_id: int, session: AsyncSession):
    await rag_chain.ainvoke({
        "chunks": chunks,
        "document_id": document_id,
        "session": session,
    })
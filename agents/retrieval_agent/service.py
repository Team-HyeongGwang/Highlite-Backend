
import os
import base64
import re
import fitz  # PyMuPDF
import asyncio
from pathlib import Path
from db.models import Document
from agents.importance_agent.schemas import ImportanceRequest
from typing import Optional
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DocumentChunk
from .schemas import PDFChunk
from agents.importance_agent.service import analyze_chunk_importance
from common.schemas import VisualCue
from ranks.service import get_ranking

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Gemini로 바꿀 수도 있음
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small", # 임베딩 모델 고민 필요
    api_key=OPENAI_API_KEY,
)

# ── PDF 청크 파싱 ────────────────────────────────────────
def parse_chunk(raw: dict) -> PDFChunk:
    raw_content = raw["content"]

    def extract_tag(tag: str) -> tuple[Optional[str], Optional[str]]:
        match = re.search(rf"\[{tag}: (.+?)(?:\s*\|\s*색상:\s*(.+?))?\]", raw_content)
        if match:
            return match.group(1).strip(), match.group(2).strip() if match.group(2) else None
        return None, None

    def has_tag(tag: str) -> bool:
        return bool(re.search(rf"\[{tag}: .+?\]", raw_content))

    handwriting, handwriting_color = extract_tag("손필기")
    highlight, highlight_color = extract_tag("형광펜")
    is_underline = has_tag("밑줄")
    is_circled = has_tag("강조")
    is_image = has_tag("이미지")

    clean_content = re.sub(r"\[(손필기|형광펜|밑줄|강조|이미지): .+?\]", "", raw_content).strip()

    # 이미지는 설명문을 content로 사용
    if is_image:
        image_match = re.search(r"\[이미지: (.+?)\]", raw_content)
        final_content = image_match.group(1).strip() if image_match else ""
    else:
        final_content = clean_content or handwriting or highlight or ""

    return PDFChunk(
        page=raw["page"],
        paragraph_index=raw["paragraph_index"],
        content=final_content,
        handwriting_color=handwriting_color,
        highlight_color=highlight_color,
        is_underline=is_underline,
        is_circled=is_circled,
        is_image=is_image,
    )


# ── DB에 Document 저장 ────────────────────────────────────────
async def init_document(input_data: dict) -> dict:
    pdf_path: Path = input_data["pdf_path"]
    
    document = Document(
        user_id=input_data["user_id"],
        group_id=input_data["group_id"],
        title=pdf_path.stem,
        doc_type="combined", 
    )
    session: AsyncSession = input_data["session"]
    session.add(document)
    await session.flush() 
    
    print(f"[1] document 생성 완료 id={document.id}")
    input_data["document_id"] = document.id  # 상태 추가
    return input_data



# ── PDF를 Image로 변환한 뒤 내용 추출 ────────────────────────────────────────
async def extract_pdf_to_raw(input_data: dict) -> dict:
    pdf_path: Path = input_data["pdf_path"]
    doc = fitz.open(str(pdf_path))
    raw_chunks = []

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=200)
        b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")

        response = await llm.ainvoke([
        SystemMessage(content=(
            "당신은 학습 교재 분석 전문가입니다. "
            "이 미지에서 모든 텍스트와 시각 정보를 빠짐없이 추출하세요. "
            "슬라이드 상단/하단의 메타 정보는 추출하지 마세요. 텍스트 외 다른 말은 절대 하지 마세요."
            "\n\n"

            "[텍스트 추출 규칙]\n"
            "1. 문단은 빈 줄(\\n\\n)로 구분하세요.\n"
            "2. 목록(bullet/번호)은 각 항목을 개별 문단으로 분리하세요. 하위 항목도 각각 독립된 문단으로 분리하세요.\n"
            "   예시:\n"
            "   * 상위 항목\n"
            "      * 하위 항목1\n"
            "      * 하위 항목2\n"
            "   → '상위 항목', '하위 항목1', '하위 항목2'를 각각 별도 문단으로 추출\n"
            "\n"
    
            "[시각 정보 추출 규칙]\n"
            "3. 형광펜으로 강조된 텍스트: [형광펜: 내용 | 색상: 색깔]\n"
            "4. 손필기로 쓰여진 텍스트: [손필기: 내용 | 색상: 색깔]\n"
            "5. 밑줄 표시된 텍스트: [밑줄: 내용]\n"
            "6. 동그라미/박스 표시된 텍스트: [강조: 내용]\n"
            "\n"
            
            "[이미지/도표 처리 규칙]\n"
            "7. 텍스트가 아닌 이미지, 그림, 도표, 그래프가 있으면 반드시 아래 형식으로 설명하세요:\n"
            "   [이미지: 해당 이미지에 대한 상세 설명]\n"
        "   예시: [이미지: 세포 분열 과정을 나타낸 그림으로, 좌측부터 간기, 분열기, 말기 순서로 표현됨]\n"
        )),
        HumanMessage(content=[
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                },
                {"type": "text", "text": f"페이지 {page_num} 텍스트를 추출해 주세요."},
            ]),
        ])

        page_text = response.content.strip()
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]

        for para_idx, content in enumerate(paragraphs, start=1):
            raw_chunks.append({
                "page": page_num,
                "paragraph_index": para_idx,
                "content": content,
            })

    doc.close()
    print(f"[2] PDF 파싱 완료 count={len(raw_chunks)}")
    input_data["raw_chunks"] = raw_chunks  # 상태 추가
    return input_data


# ── 생 데이터를 PDFChunk로 내용 파싱 ────────────────────────────────────────
async def process_raw_chunks(input_data: dict) -> dict:
    raw_chunks = input_data["raw_chunks"]
    
    chunks = [parse_chunk(raw) for raw in raw_chunks]
    print(f"[3] 청크 변환 완료 count={len(chunks)}")
    
    input_data["chunks"] = chunks  # 이제 아래의 rag_chain 단계들이 이 chunks를 소모합니다.
    return input_data


# ── 로깅 단계 ────────────────────────────────────────
async def receive_chunks(input_data: dict) -> dict:
    chunks = input_data["chunks"]
    print(f"\n[rag_agent] 받은 청크 수: {len(chunks)}")
    for chunk in chunks:
        print(f"  - page={chunk.page} para={chunk.paragraph_index} content={chunk.content[:30]}...")
    return input_data


# ── 임베딩 계산 ────────────────────────────────────────
async def embed_chunks(input: dict) -> dict:
    chunks: list[PDFChunk] = input["chunks"]
    texts = [chunk.content for chunk in chunks]

    embeddings = await embeddings_model.aembed_documents(texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    return input


# ── 임베딩 값 DB에 저장 ────────────────────────────────────────
async def save_embeddings_to_db(input: dict) -> dict:
    chunks: list[PDFChunk] = input["chunks"]
    document_id: int = input["document_id"]
    session: AsyncSession = input["session"]

    for chunk in chunks:
        db_chunk = DocumentChunk(
            document_id=document_id,
            page_number=chunk.page,
            original_text=chunk.content,
            embedding=chunk.embedding,
            meta_data={
                "paragraph_index": chunk.paragraph_index,
                "handwriting_color": chunk.handwriting_color,
                "highlight_color": chunk.highlight_color,
                "is_underline": chunk.is_underline,
                "is_circled": chunk.is_circled,
                "is_image": chunk.is_image,
            },
        )
        
        session.add(db_chunk)
        await session.flush() 
        chunk.db_id = db_chunk.id  
        
    return input


# ── importance_agent에게 값 전달 ──────────────────────────────────────────
async def send_to_importance_agent(input: dict) -> dict:
    chunks: list[PDFChunk] = input["chunks"]
    session: AsyncSession = input["session"]
    user_id: int = input["user_id"]
    group_id = str(input["group_id"])
    
    doc_type = "pdf"

    # DB에서 사용자 ranking 조회
    ranking = await get_ranking(user_id, session)
    highlighter_ranking = ranking["highlighter_ranking"] if ranking else {}
    pen_ranking = ranking["pen_ranking"] if ranking else {}

    tasks = []

    for chunk in chunks:
        visual_cues: list[VisualCue] = []

        if chunk.highlight_color:
            visual_cues.append(VisualCue(
                type="highlight",
                color=chunk.highlight_color,
                target_text=chunk.content,  # ✅ chunk.content로 대통합!
            ))

        if chunk.handwriting_color:
            visual_cues.append(VisualCue(
                type="pen",
                color=chunk.handwriting_color,
                target_text=chunk.content,
            ))

        request = ImportanceRequest(
            group_id=group_id,
            chunk_id=chunk.db_id,
            doc_type=doc_type,
            original_text=chunk.content,
            meta_data=visual_cues,
            highlighter_ranking=highlighter_ranking,
            pen_ranking=pen_ranking,
        )

        tasks.append(analyze_chunk_importance(request, session))
        
        # 담아둔 모든 중요도 분석 작업을 동시에 병렬(Concurrent)로 실행
    print(f"\n[Master Pipeline] 총 {len(tasks)}개의 청크 중요도 분석을 동시에 시작합니다... 🚀")
    
    # asyncio.gather가 모든 대기 작업을 한 번에 쏘고 결과를 다 모아서 가져옴
    await asyncio.gather(*tasks)

    await session.commit()
    
    print(f"[Master Pipeline] 모든 청크의 중요도 분석 및 DB 저장이 완료되었습니다! 🎯")
        
    return input


# ── 체인 조립 ──────────────────────────────────────────
pdf_pipeline_chain = (
    RunnableLambda(init_document)
    | RunnableLambda(extract_pdf_to_raw)
    | RunnableLambda(process_raw_chunks)
    | RunnableLambda(receive_chunks)
    | RunnableLambda(embed_chunks)
    | RunnableLambda(save_embeddings_to_db)
    | RunnableLambda(send_to_importance_agent)
)

# 외부(라우터)에서 호출할 메인 파이프라인 구동 함수
async def run_pdf_pipeline(
    pdf_path: Path,
    user_id: int,
    group_id: str,
    session: AsyncSession,
) -> None:
    try:
        print(f"\n[Master Pipeline] '{pdf_path.name}' 파이프라인 구동을 시작합니다... 🚀")
        
        await pdf_pipeline_chain.ainvoke({
            "pdf_path": pdf_path,
            "user_id": user_id,
            "group_id": group_id,
            "session": session,
        })
        print(f"[Success] 전체 PDF 파이프라인 체인이 에러 없이 완주했습니다! 🎯\n")
    except Exception as e:
        print(f"[Error] 파이프라인 수행 중 에러 발생: {e}")
        raise
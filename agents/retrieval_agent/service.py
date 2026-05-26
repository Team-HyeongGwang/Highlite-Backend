
import os
import base64
import fitz  # PyMuPDF
import asyncio
from pathlib import Path
from db.models import Document
from agents.importance_agent.schemas import ImportanceRequest
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DocumentChunk
from .schemas import PDFChunk, PageOutput
from agents.importance_agent.service import analyze_chunk_importance
from common.schemas import VisualCue
from ranks.service import get_ranking

from typing import List

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

# ── PageOutput → PDFChunk 리스트 변환 ─────────────────────────
def to_pdf_chunks(page_num: int, page_output: PageOutput) -> List[PDFChunk]:
    chunks = []
    
    for paragraph_index, chunk in enumerate(page_output.chunks):
        chunks.append(PDFChunk(
            page=page_num,
            paragraph_index=paragraph_index,
            content=chunk.content,
            handwriting_color=chunk.handwriting_color,
            highlight_color=chunk.highlight_color,
            is_underline=chunk.is_underline,
            is_circled=chunk.is_circled,
            is_image=chunk.is_image,
        ))
        
    return chunks

# ── PDF를 Image로 변환한 뒤 내용 추출 ────────────────────────────────────────
async def extract_pdf_to_raw(input_data: dict) -> dict:
    pdf_path: Path = input_data["pdf_path"]
    doc = fitz.open(str(pdf_path))
    raw_chunks = []

    structured_llm = llm.with_structured_output(PageOutput)

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=200)
        b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
        
        page_output: PageOutput = await structured_llm.ainvoke([
            SystemMessage(content=(
                "당신은 강의 자료를 분석하는 AI입니다.\n\n"
                
                "[가장 중요한 지시사항]\n"
                "   - 텍스트 추출 전, 가장 먼저 슬라이드의 전체적인 맥락과 구성, 전체 흐름을 이해하세요.\n"
                "   - 해당 단계에서 이해한 슬라이드의 맥락, 구성을 청킹과 텍스트, 이미지 처리에서도 사용하세요.\n"
                "   - 해당 과정을 진행하면서, 스스로가 어떠한 과정을 거쳤는지에 대한 내용을 적지마세요. (ex: 반성중)"
                "   - 슬라이드의 전체 흐름을 이해하되, 슬라이드 내 정보만을 활용하세요.\n"

                "1. 청킹(Chunking)\n"
                "   - 청킹 전에 앞서 정리했던 슬라이드의 맥락, 구성을 활용하세요.\n"
                "   - 적당히 문단 단위로 진행하세요.\n"
                "   - 단어 단위나 너무 짧은 줄 단위로 잘게 끊지 마세요. 문맥이 이어지는 한 덩어리로 묶어주세요.\n"
                "   - 제목란과 본문란을 구분하세요.\n"
                "   - 슬라이드 상하단의 페이지 번호, 강의명 같은 불필요한 정보는 버리세요.\n\n"

                "2. 텍스트와 이미지, 필기를 자연스럽게 처리하세요.\n"
                "   - 청킹 전에 앞서 정리했던 슬라이드의 맥락, 구성을 활용하세요.\n"
                "   - [텍스트]: '실제로 적혀있는 글씨'만 그대로 가져와서 'content'에 적습니다.\n"
                "   - [이미지]: 'is_image=true'로 설정하고, 이미지의 객관적인 사실만 'content'에 설명합니다.\n"
                "   - [도표/이미지 속 글자]: 그림이나 도표(그래프, 다이어그램 등) '안에' 포함된 글씨들은 절대 개별 텍스트 청크로 분리해서 따로 빼내지 마세요!!\n"
                "   - [도표/이미지 주변, 속 손필기나 형광펜]: 이미지의 객관적 사실과 함께 손필기나 형광펜으로 강조된 내용을 엮어서 'content'에 포함시킵니다.\n"
                "   - [필기/형광펜]: 메타데이터 플래그를 켭니다.\n"
                "   - [필기 색상]: 형광펜과 필기 색깔이 겹쳐있는 경우에는 둘을 분리하여 인식하세요.\n"
            )),
                        HumanMessage(content=[
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                },
                {"type": "text", "text": "슬라이드 내용을 추출해 주세요."},
            ]),
        ])


        raw_chunks.extend(to_pdf_chunks(page_num, page_output))


    doc.close()
    print(f"[2] PDF 파싱 완료 count={len(raw_chunks)}")
    input_data["chunks"] = raw_chunks
    return input_data


# ── 생 데이터를 PDFChunk로 내용 파싱 ────────────────────────────────────────
async def process_raw_chunks(input_data: dict) -> dict:
    chunks = input_data["raw_chunks"]  # PDFChunk 타입 
    print(f"[3] 청크 변환 완료 count={len(chunks)}")
    input_data["chunks"] = chunks
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
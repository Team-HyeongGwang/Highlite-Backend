import os
import base64
import re
import fitz  # PyMuPDF
import uuid
import json
from typing import TypedDict
import asyncio
from pathlib import Path
from db.models import Document
from agents.importance_agent.schemas import ImportanceRequest
from typing import Optional
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small", # 임베딩 모델 고민 필요
    api_key=OPENAI_API_KEY,
)

# ── PDF 청크 파싱 ────────────────────────────────────────
def parse_chunk(raw: dict, pdf_name: str) -> PDFChunk:
    # 1. 내용(content)을 안전하게 문자열로 변환
    raw_content = str(raw.get("content", ""))

    # 2. 제미나이가 JSON으로 예쁘게 뽑아준 색상 정보 바로 가져오기
    handwriting_color = raw.get("handwriting_color")
    highlight_color = raw.get("highlight_color")

    # 3. 혹시나 내용에 남아있을 수 있는 [태그] 찌꺼기 깔끔하게 청소
    clean_content = re.sub(r"\[(손필기|형광펜): .+?\]", "", raw_content).strip()

    return PDFChunk(
        pdf_name=pdf_name,
        page=raw.get("page", 1),
        paragraph_index=raw.get("paragraph_index", 1),
        content=clean_content or raw_content,
        handwriting_color=handwriting_color,
        highlight_color=highlight_color,
    )


# ── DB에 Document 저장 ────────────────────────────────────────
async def init_document(input_data: dict) -> dict:
    pdf_path: Path = input_data["pdf_path"]
    4
    document = Document(
        user_id=input_data["user_id"],
        group_id=str(input_data["group_id"]),
        title=pdf_path.stem,
        doc_type="combined", 
    )
    session: AsyncSession = input_data["session"]
    session.add(document)
    await session.flush() 
    
    print(f"[1] document 생성 완료 id={document.id}")
    input_data["document_id"] = document.id
    return input_data



async def extract_pdf_to_raw(input_data: dict) -> dict:
    pdf_path: Path = input_data["pdf_path"]
    raw_chunks = []
    
    pages_data = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=150) # 유료니까 해상도 150으로 선명하게!
            b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
            pages_data.append((page_num, b64))
            
    print(f"\n[2-1] 총 {len(pages_data)}장 이미지 변환 완료. 제미나이 분석 시작 (유료 모드 풀가동 🚀)...")

    # ⭐️ 1. 1장씩만 깔끔하게 분석하는 함수
    async def process_single_page(page_num, b64):
        response = await llm.ainvoke([
            SystemMessage(content=("""
            당신은 강의 자료(PDF 슬라이드)를 분석하여 RAG 시스템에 활용될 청크(Chunk) 단위 데이터로 변환하는 AI입니다.

            [기본 원칙]
            1. 텍스트 추출 전, 슬라이드의 전체적인 맥락과 흐름을 먼저 파악하고 이를 바탕으로 청킹(Chunking)하세요.
            2. 당신의 분석 과정, 생각, 부연 설명은 절대 출력하지 마세요. 오직 요청된 JSON 배열 형태만 반환해야 합니다.
            3. 슬라이드 상/하단의 페이지 번호, 강의명 등 본문과 무관한 텍스트는 추출에서 제외하세요.

            [데이터 처리 가이드]
            1. 청킹(Chunking) 기준
            - '의미상 이어지는 단일 개념'이나 '문단' 단위로 묶어주세요. 단어나 짧은 줄 단위로 파편화하지 마세요.
            - 위에서부터 아래로 자연스러운 흐름에 따라 `paragraph_index`를 1부터 순차적으로 부여하세요.

            2. 요소별 추출 규칙 (content 작성)
            - [일반 텍스트]: 슬라이드에 적힌 실제 텍스트를 문맥에 맞게 묶어 `content`에 작성합니다.
            - [이미지/도표]: 그림이나 도표 안에 포함된 글자들을 따로 떼어내지 마세요. 도표의 의미와 포함된 텍스트를 종합하여 객관적인 묘사로 `content`에 작성합니다.
            - [필기/형광펜]: 슬라이드에 손글씨(handwriting)나 형광펜(highlight)이 칠해져 있다면, 그 텍스트나 의미를 `content`에 자연스럽게 포함시킵니다. 그리고 반드시 해당 색상을 별도 필드에 명시하세요.

            [출력 형식]
            반드시 아래 JSON 배열(List) 형식으로만 응답하세요. 다른 텍스트는 금지입니다.
            [
              {
                  "paragraph_index": 1,
                  "content": "추출된 텍스트 내용 또는 이미지/도표에 대한 설명",
                  "handwriting_color": "blue",
                  "highlight_color": "yellow"
              }
            ]
            """)),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": f"페이지 {page_num} 텍스트를 추출해 주세요."}
            ])
        ])
        
        page_chunks = []
        try:
            raw_content = response.content
            if isinstance(raw_content, list):
                raw_text = "".join([str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in raw_content])
            else:
                raw_text = str(raw_content)
                
            raw_text = raw_text.strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            paragraphs = json.loads(raw_text)
            for para_idx, item in enumerate(paragraphs, start=1):
                if isinstance(item, dict):
                    page_chunks.append({
                        "page": page_num,
                        "paragraph_index": item.get("paragraph_index", para_idx),
                        "content": item.get("content", ""),
                        "handwriting_color": item.get("handwriting_color"),
                        "highlight_color": item.get("highlight_color"),
                    })
        except Exception as e:
            print(f"[Error] {page_num} 페이지 파싱 실패: {e}")
            
        print(f"   -> {page_num} 페이지 추출 완료! ⚡️")
        return page_chunks

    semaphore = asyncio.Semaphore(50)
    
    async def sem_process(page_num, b64):
        async with semaphore:
            return await process_single_page(page_num, b64)

    tasks = [sem_process(page_num, b64) for page_num, b64 in pages_data]
    results = await asyncio.gather(*tasks)
    
    for res in results:
        if res:
            raw_chunks.extend(res)

    raw_chunks.sort(key=lambda x: (x.get("page", 0), x.get("paragraph_index", 0)))
    
    print(f"\n[2-2] PDF 파싱 완료! 총 청크 개수: {len(raw_chunks)}")
    input_data["raw_chunks"] = raw_chunks  
    return input_data


# ── 생 데이터를 PDFChunk로 내용 파싱 ────────────────────────────────────────
async def process_raw_chunks(input_data: dict) -> dict:
    raw_chunks = input_data["raw_chunks"]
    pdf_name = input_data["pdf_path"].stem
    
    chunks = [parse_chunk(raw, pdf_name) for raw in raw_chunks]
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
    document_id: str = str(input["document_id"])
    session: AsyncSession = input["session"]

    db_chunks = []
    
    for chunk in chunks:
        db_chunk = DocumentChunk(
            document_id=document_id,
            page_number=chunk.page,
            original_text=chunk.content,
            embedding=chunk.embedding,
            meta_data={
                "pdf_name": chunk.pdf_name,
                "paragraph_index": chunk.paragraph_index,
                "handwriting_color": chunk.handwriting_color,
                "highlight_color": chunk.highlight_color,
            },
        )
        session.add(db_chunk)
        db_chunks.append(db_chunk)
        
    await session.flush() 
    
    for chunk, db_chunk in zip(chunks, db_chunks):
        chunk.db_id = db_chunk.id  
        
    return input


# ── importance_agent에게 값 전달 ──────────────────────────────────────────
async def send_to_importance_agent(input: dict) -> dict:
    chunks: list[PDFChunk] = input["chunks"]
    session: AsyncSession = input["session"]
    user_id: int = input["user_id"]
    group_id = str(input["group_id"])
    
    doc_type = "pdf"

    ranking = await get_ranking(user_id, session)
    highlighter_ranking = (ranking["highlighter_ranking"] or {}) if ranking else {}
    pen_ranking = (ranking["pen_ranking"] or {}) if ranking else {}

    print(f"\n[Master Pipeline] 총 {len(chunks)}개의 청크 중요도 분석 시작... (10개씩 API 호출 후 일괄 DB 저장) 🚀")

    semaphore = asyncio.Semaphore(50)

    async def sem_analyze(req):
        async with semaphore:
            return await analyze_chunk_importance(req)

    tasks = []

    for chunk in chunks:
        visual_cues: list[VisualCue] = []

        if chunk.highlight_color:
            visual_cues.append(VisualCue(type="highlight", color=chunk.highlight_color, target_text=chunk.content))
        if chunk.handwriting_color:
            visual_cues.append(VisualCue(type="pen", color=chunk.handwriting_color, target_text=chunk.content))

        request = ImportanceRequest(
            group_id=group_id,
            chunk_id=chunk.db_id,
            doc_type=doc_type,
            original_text=chunk.content,
            meta_data=visual_cues,
            highlighter_ranking=highlighter_ranking,
            pen_ranking=pen_ranking,
        )

        tasks.append(sem_analyze(request))
        
    # ⭐️ 1. 에이전트들이 123개 청크를 병렬로 마구마구 분석하고 결과만 배열로 모아옴
    llm_results = await asyncio.gather(*tasks)

    # ⭐️ 2. 모아온 결과(123개)를 안전하게 순차적으로 DB에 삽입 (충돌 완벽 방지)
    for db_result in llm_results:
        session.add(db_result)

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
        
        result = await pdf_pipeline_chain.ainvoke({
            "pdf_path": pdf_path,
            "user_id": user_id,
            "group_id": group_id,
            "session": session,
        })

        print(f"[Success] 전체 PDF 파이프라인 체인이 에러 없이 완주했습니다! 🎯\n")
        return [result["document_id"]] # Response JSON에 출력 결과 담음
    except Exception as e:
        print(f"[Error] 파이프라인 수행 중 에러 발생: {e}")
        raise
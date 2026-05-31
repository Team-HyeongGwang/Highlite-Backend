import json
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .schemas import ImportanceRequest, ImportanceResponse
import db.models as models

async def analyze_chunk_importance(request: ImportanceRequest) -> models.ImportanceResult:
    """
    [1타 강사 AI] OpenAI GPT-4o-mini의 미친 속도 + 타임아웃 방어막이 결합된 최종 버전!
    """
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    structured_llm = llm.with_structured_output(ImportanceResponse)
    
    system_prompt = """
    당신은 학습자의 교재(교과서, 전공서적, 수험서 등) 및 필기 데이터를 분석하여 '시험 출제 확률 및 핵심 개념(중요도)'을 0.0에서 10.0 사이의 점수로 평가하는 1타 강사 AI 에이전트입니다.    
    
    [핵심 평가 로직: 사용자 가중치 탄력적 적용]
    사용자는 펜과 형광펜의 중요도 순위를 최소 1개에서 최대 3개까지 자유롭게 설정할 수 있습니다. 
    입력된 `highlighter_ranking`과 `pen_ranking` 데이터를 확인하고, '존재하는 순위'에 대해서만 아래의 가중치를 엄격하게 적용하세요.

    * 1순위로 지정된 색상 발견 시: 가중치 2.0 (점수 대폭 상승, 9.0 ~ 10.0)
    * 2순위로 지정된 색상 발견 시: 가중치 1.5 (점수 중폭 상승, 7.0 ~ 8.9)
    * 3순위로 지정된 색상 발견 시: 가중치 1.0 (점수 소폭 상승, 4.0 ~ 6.9)
    * 순위에 등록되지 않은 색상이거나 시각 정보가 없는 경우: 철저히 텍스트의 문맥(개념, 정의 등)만으로 기본 점수 부여 (0.0 ~ 5.0)
    
    [입력 데이터]
    - 사용자 형광펜 랭킹: {highlighter_ranking}
    - 사용자 펜 랭킹: {pen_ranking}
    """
    
    user_prompt = """
    아래 문서 조각(Chunk)과 시각 정보(Visual Cues)를 분석하여 중요도를 평가해 주세요.
    
    [원본 텍스트]
    {original_text}
    
    [시각 정보 (필기 데이터)]
    {meta_data}
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    chain = prompt_template | structured_llm
    
    MAX_RETRIES = 3
    response = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.wait_for(
                chain.ainvoke({
                    "highlighter_ranking": json.dumps(request.highlighter_ranking, ensure_ascii=False),
                    "pen_ranking": json.dumps(request.pen_ranking, ensure_ascii=False),
                    "original_text": request.original_text,
                    "meta_data": json.dumps([cue.model_dump() for cue in request.meta_data], ensure_ascii=False)
                }),
                timeout=20.0 
            )
            break 
            
        except asyncio.TimeoutError:
            print(f"[Warning] 청크 {request.chunk_id} 응답 잠수! ({attempt+1}/{MAX_RETRIES}) 즉시 재시도 🔄")
        except Exception as e:
            print(f"[Warning] 청크 {request.chunk_id} 통신 에러. 1초 후 재시도 🔄")
            await asyncio.sleep(1)

    if not response:
        print(f"[Error] 청크 {request.chunk_id} 최종 실패! 기본값(0점)으로 처리하고 넘어갑니다. 😭")
        response = ImportanceResponse(score=0.0, reasoning="API 분석 실패", summary="분석 실패", keywords=[])

    db_result = models.ImportanceResult(
        chunk_id=request.chunk_id,
        score=response.score,
        reasoning=response.reasoning,
        summary=response.summary,
        keywords=response.keywords 
    )
    
    return db_result
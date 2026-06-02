import json
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .schemas import ImportanceRequest, ImportanceResponse
import db.models as models

async def analyze_chunk_importance(request: ImportanceRequest) -> models.ImportanceResult:
    """
    [1타 강사 AI] 파이썬 파서가 넘겨준 텍스트와 시각 정보를 바탕으로, 
    오직 중요도 점수 계산과 키워드 추출에만 집중하는 최적화된 LLM 호출 함수입니다.
    """
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    structured_llm = llm.with_structured_output(ImportanceResponse)
    
    system_prompt = """
    당신은 학습자의 교재(교과서, 전공서적, 수험서 등) 및 필기 데이터를 분석하여 '시험 출제 확률 및 핵심 개념(중요도)'을 0.0에서 10.0 사이의 점수로 평가하는 1타 강사 AI 에이전트입니다.    
    
    [특수 강조 태그(Tag) 위계 및 분석 가이드]
    입력된 텍스트(`original_text`) 내에 사용자가 직접 표시한 특수 태그가 존재할 경우, 태그의 '의미적 위계'에 따라 차등적인 중요도 기본 점수(Base Score)를 산정하세요.
    - ⭐️ 최상위 강조 태그 ([별표], [중요], [시험], [핵심] 등 이와 유사한 의미를 띠는 모든 태그) : 사용자가 '시험 출제'나 '최고 중요도'를 확신하고 표시한 핵심 정보입니다. (기본 점수 대폭 상승, 7.0 ~ 9.0)
    - 🖍️ 시각적 강조 태그 ([동그라미], [네모], [밑줄]) : 사용자가 문맥상 집중을 위해 시각적으로 표시한 강조 정보입니다. (기본 점수 소폭~중폭 상승, 5.0 ~ 7.0)
    - 📝 부연 설명 태그 ([필기: 내용]) : 본문을 보충하는 단순 메모입니다. 이 태그가 존재한다는 이유만으로 중요도를 높이지 마세요. 철저히 일반 인쇄 텍스트와 동등하게 취급하여 전체 문맥을 이해하는 용도로만 활용하세요.

    [핵심 평가 로직: 사용자 가중치 탄력적 적용]
    사용자는 펜과 형광펜의 중요도 순위를 자유롭게 설정(최소 0개 ~ 최대 3개)할 수 있습니다. 
    입력된 `highlighter_ranking`과 `pen_ranking` 데이터를 확인하고, '존재하는 순위'에 대해서만 아래의 가중치를 엄격하게 적용하여 최종 점수를 결정하세요.

    * 1순위로 지정된 색상 발견 시: 가중치 2.0 (점수 대폭 상승, 9.0 ~ 10.0) -> [최상위 강조 태그]와 결합 시 무조건 10점 만점!
    * 2순위로 지정된 색상 발견 시: 가중치 1.5 (점수 중폭 상승, 7.0 ~ 8.9)
    * 3순위로 지정된 색상 발견 시: 가중치 1.0 (점수 소폭 상승, 4.0 ~ 6.9)
    * 순위에 등록되지 않은 색상이거나 시각 정보가 없는 경우: 텍스트 문맥과 [특수 강조 태그]의 유무만으로 기본 점수 부여
      - 일반 텍스트: 0.0 ~ 5.0
      - [시각적 강조 태그] 포함 시: 최대 7.0까지 허용
      - [최상위 강조 태그] 포함 시: 최대 8.5까지 허용 (순위가 없어도 기호 자체의 압도적 중요성 인정)
    
    ⛔️ [주의 사항: 순위 미지정 색상 처리]
    시각 정보(meta_data)에 필기(pen)나 형광펜(highlight) 색상이 존재하더라도, 사용자가 전달한 `ranking` 딕셔너리에 해당 색상이 명시되어 있지 않다면 '색상으로 인한 추가 가중치(1순위~3순위)'는 절대 부여하지 마세요. 
    단, 해당 색상으로 작성된 특수 태그가 존재한다면 위에 명시된 '태그의 위계(최상위 > 시각적)'에 따라 텍스트 기본 점수에 충실히 반영해야 합니다.
    
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
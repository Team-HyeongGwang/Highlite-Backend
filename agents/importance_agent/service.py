import json
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .schemas import ImportanceRequest, ImportanceResponse
import db.models as models

async def analyze_chunk_importance(request: ImportanceRequest, db: AsyncSession) -> ImportanceResponse:
    """
    [1타 강사 AI] 파이썬 파서가 넘겨준 텍스트와 시각 정보를 바탕으로, 
    오직 중요도 점수 계산과 키워드 추출에만 집중하는 최적화된 LLM 호출 함수입니다.
    """
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # LLM output 양식
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
    - 사용자 형광펜 랭킹 (1~3개 유동적): {highlighter_ranking}
    - 사용자 펜 랭킹 (1~3개 유동적): {pen_ranking}
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
    
    response: ImportanceResponse = chain.invoke({
        "highlighter_ranking": json.dumps(request.highlighter_ranking, ensure_ascii=False),
        "pen_ranking": json.dumps(request.pen_ranking, ensure_ascii=False),
        "original_text": request.original_text,
        "meta_data": json.dumps([cue.model_dump() for cue in request.meta_data], ensure_ascii=False)
    })
    
    db_result = models.ImportanceResult(
        chunk_id=request.chunk_id,
        score=response.score,
        reasoning=response.reasoning,
        summary=response.summary,
        keywords=response.keywords 
    )
    
    db.add(db_result)  
    # await db.commit() 
    
    return response
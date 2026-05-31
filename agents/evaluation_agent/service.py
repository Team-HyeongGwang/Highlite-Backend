import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from db.models import Question, ImportanceResult, DocumentChunk
# 보내주신 스키마 경로에 맞춰 임포트
from agents.evaluation_agent.schemas import (
    QuestionReviewRequest, QuestionReviewResponse,
)

# 검수용 LLM 설정 (일관된 검수를 위해 temperature=0 고정)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

REQUIRED_OPTION_KEYS = {"①", "②", "③", "④"}

async def review(req: QuestionReviewRequest) -> QuestionReviewResponse:
    # 1. 객관식 문제 선지 완성도 사전 검증 (LLM 호출 전)
    if req.options is not None:
        missing_keys = REQUIRED_OPTION_KEYS - set(req.options.keys())
        if missing_keys:
            missing_str = ", ".join(sorted(missing_keys))
            return QuestionReviewResponse(
                is_approved=False,
                quality_score=2,
                feedback=f"선지 누락: {missing_str} 보기가 생성되지 않았습니다. 4개 선지(①②③④)가 모두 필요합니다.",
                suggested_revision_text=None,
                suggested_revision_options=None,
            )

    # 2. Question Agent가 넘겨준 보기가 존재하면 텍스트 포맷으로 변환
    options_text = ""
    if req.options:
        options_text = "\n".join([f"  {k}: {v}" for k, v in req.options.items()])

    # 2. 고정된 Question Agent의 데이터 규격에 100% 동기화되도록 프롬프트 구성
    prompt_content = f"""
당신은 대학교 시험 문제 검수 전문가입니다. Question Agent가 생성한 문제를 원본 텍스트에 근거하여 엄격하게 검수해주세요.

[원본 텍스트]
{req.source_chunk_text}

[검수할 문제]
문제: {req.question_text}
보기:
{options_text}
정답: {req.answer}
해설: {req.explanation}

검수 기준:
1. 원본 텍스트에 없는 내용이나 사실과 다른 왜곡된 내용이 포함되지 않았는가?
2. 정답이 혼동 없이 명확하게 하나뿐인가?
3. 오답 보기들이 그럴싸하면서도 학술적으로 명확히 틀렸는가?
4. 해설이 원문 근거를 포함하여 정답을 올바르고 쉽게 설명하는가?

반드시 아래에 지정된 JSON 형식으로만 응답하세요. 마크다운 기호(```json)나 별도의 인사말, 부연 설명은 절대 포함하지 마세요.

{{
  "is_approved": true 또는 false (원문에 완벽히 충실하고 결함이 없다면 true, 수정이 필요하면 false),
  "quality_score": 1~10 사이 정수 (문제의 완성도 및 적합성 점수),
  "feedback": "반려 사유 또는 '이상 없음'",
  "suggested_revision_text": null 또는 "수정한 문제 지문 (is_approved가 false일 때만 작성, 정상적이면 null)",
  "suggested_revision_options": null 또는 {{"①": "...", "②": "...", "③": "...", "④": "..."}} (is_approved가 false이고 객관식 문제일 때만 반드시 이 원문자 키 형식으로 작성, 아니면 null)
}}
"""

    # 3. LLM 호출
    response = await llm.ainvoke([HumanMessage(content=prompt_content)])

    # 4. LLM 응답 텍스트에서 불필요한 마크다운 펜스 제거 및 공백 정제
    clean_content = response.content.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean_content)

    # 5. QuestionReviewResponse 스키마에 맞춰 규격화된 데이터 반환
    return QuestionReviewResponse(
        is_approved=bool(parsed["is_approved"]),
        quality_score=int(parsed["quality_score"]),
        feedback=parsed["feedback"],
        suggested_revision_text=parsed.get("suggested_revision_text"),
        suggested_revision_options=parsed.get("suggested_revision_options"),
    )
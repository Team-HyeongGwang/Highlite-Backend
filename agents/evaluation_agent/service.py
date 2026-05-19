import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from db.models import Question, ImportanceResult, DocumentChunk
from agents.evaluation_agent.schemas import (
    QuestionReviewRequest, QuestionReviewResponse,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_FEEDBACK_INSTRUCTION = {
    FeedbackType.wrong_answer:     "정답이 틀렸습니다. 정답이 명확한 새 문제를 출제하세요.",
    FeedbackType.ambiguous:        "문제가 애매합니다. 더 명확하고 구체적인 문제로 재출제하세요.",
    FeedbackType.not_in_source:    "원본 텍스트에 없는 내용이 포함됐습니다. 원본에만 근거해 재출제하세요.",
    FeedbackType.multiple_correct: "복수 정답이 가능합니다. 정답이 하나만 되도록 재출제하세요.",
}


async def review(req: QuestionReviewRequest) -> QuestionReviewResponse:
    options_text = ""
    if req.options:
        options_text = "\n".join([f"  {k}: {v}" for k, v in req.options.items()])

    response = await llm.ainvoke([HumanMessage(content=f"""
당신은 시험 문제 검수 전문가입니다.

[원본 텍스트]
{req.source_chunk_text}

[검수할 문제]
문제: {req.question_text}
보기:
{options_text}
정답: {req.answer}
해설: {req.explanation}

검수 기준:
1. 원본 텍스트에 없는 내용이 포함되지 않았는가?
2. 정답이 명확하게 하나인가?
3. 오답 보기가 명확히 틀린가?
4. 해설이 정답을 올바르게 설명하는가?

JSON 형식으로만 응답하세요:
{{
  "is_approved": true 또는 false,
  "quality_score": 1~10 사이 정수,
  "feedback": "반려 사유 또는 '이상 없음'",
  "suggested_revision_text": null 또는 "수정 제안 문제 지문",
  "suggested_revision_options": null 또는 {{"A": "...", "B": "...", "C": "...", "D": "..."}}
}}
""")])

    parsed = json.loads(response.content)

    return QuestionReviewResponse(
        is_approved=parsed["is_approved"],
        quality_score=parsed["quality_score"],
        feedback=parsed["feedback"],
        suggested_revision_text=parsed.get("suggested_revision_text"),
        suggested_revision_options=parsed.get("suggested_revision_options"),
    )



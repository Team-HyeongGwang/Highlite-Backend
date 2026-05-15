# agents/personalized_agent/agent.py

import json
from collections import Counter, defaultdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from db.models import Question, Document
from agents.personalized_agent.schemas import PersonalizationResponse

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

_session_log: dict = defaultdict(lambda: defaultdict(list))

_DIFFICULTY_DOWN = {
    "hard": "medium",
    "medium": "easy",
    "easy": "easy",
}


async def record_answer(user_id: int, group_id: str, question_id: int, user_answer: str, is_correct: bool, db: AsyncSession):
    question = await db.get(Question, question_id)
    if not question:
        return None

    _session_log[user_id][group_id].append({
        "question_type": question.question_type,
        "difficulty": question.difficulty,
        "is_correct": is_correct,
    })
    return True


def compute_bias(user_id: int, group_id: str) -> list:
    log = _session_log[user_id][group_id]
    if not log:
        return []

    wrong_counts: Counter = Counter()
    total_counts: Counter = Counter()

    for entry in log:
        key = (entry["question_type"], entry["difficulty"])
        total_counts[key] += 1
        if not entry["is_correct"]:
            wrong_counts[key] += 1

    bias = [
        {
            "question_type": k[0],
            "current_difficulty": k[1],
            "recommended_difficulty": _DIFFICULTY_DOWN.get(k[1], "easy"),  # 한 단계 낮춤
            "wrong_rate": round(wrong_counts[k] / total_counts[k], 2),
            "total_attempts": total_counts[k],
        }
        for k in total_counts
        if wrong_counts[k] > 0  # 틀린 것만 bias에 포함
    ]
    bias.sort(key=lambda x: x["wrong_rate"], reverse=True)
    return bias


async def analyze_weakness(user_id: int, group_id: str, db: AsyncSession) -> PersonalizationResponse:
    log = _session_log[user_id][group_id]

    wrong_counts: Counter = Counter()
    total_counts: Counter = Counter()
    for entry in log:
        key = (entry["question_type"], entry["difficulty"])
        total_counts[key] += 1
        if not entry["is_correct"]:
            wrong_counts[key] += 1

    stats_text = "\n".join([
        f"- 유형: {k[0]}, 난이도: {k[1]} → {wrong_counts[k]}/{total_counts[k]}개 틀림"
        for k in total_counts
    ])

    doc_result = await db.execute(
        select(Document.title).where(Document.group_id == group_id).limit(1)
    )
    doc_title = doc_result.scalar() or group_id

    response = await llm.ainvoke([HumanMessage(content=f"""
학생의 문제 풀이 통계입니다:
{stats_text}

아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{
  "weakness_concepts": ["반복해서 틀리는 개념1", "개념2", "개념3"],
  "personalized_advice": "맞춤형 학습 방향 1문단",
  "next_recommendation": ["다음에 집중할 키워드1", "키워드2", "키워드3"]
}}
""")])

    parsed = json.loads(response.content)

    return PersonalizationResponse(
        weakness_concepts=parsed["weakness_concepts"],
        weakness_source_file=doc_title,
        personalized_advice=parsed["personalized_advice"],
        next_recommendation=parsed["next_recommendation"],
    )

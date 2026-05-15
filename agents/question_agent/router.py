from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import random, json, anthropic, math, asyncio, httpx, openai

from db.database import get_db
from db.models import Question, ImportanceResult, DocumentChunk, Document, User
from agents.question_agent.schemas import (
    QuestionGenerateRequest,
    QuestionGenerateResponse,
    QuestionItem,
    RegenerateRequest,
    RegenerateResponse,
)

router = APIRouter()
claude_client = anthropic.AsyncAnthropic()
gpt_client = openai.AsyncOpenAI()

EVALUATION_URL = "http://127.0.0.1:8000/evaluation/review"  # 서형이 완성되면 여기만 수정

# ────────────────────────────────────────
# 순위 결정
# ────────────────────────────────────────
def get_priority(meta_data: list, highlighter_ranking: dict, pen_ranking: dict) -> int:
    best = 99
    for cue in meta_data or []:
        color = cue.get("color") if isinstance(cue, dict) else cue.color
        cue_type = cue.get("type") if isinstance(cue, dict) else cue.type
        if cue_type == "highlight":
            rank = highlighter_ranking.get(color, 99)
        elif cue_type == "pen":
            rank = pen_ranking.get(color, 99)
        else:
            continue
        best = min(best, rank)
    return best if best != 99 else 3

# ────────────────────────────────────────
# 문제 유형 결정
# ────────────────────────────────────────
def get_question_type(priority: int) -> str:
    if priority == 1:
        return random.choice(["multiple_choice", "ox", "fill_in_the_blank"])
    else:
        return random.choice(["multiple_choice", "ox"])

# ────────────────────────────────────────
# 출제 수 분배 (1순위 50% / 2순위 30% / 3순위 20%)
# ────────────────────────────────────────
def distribute_counts(total: int, chunks_by_priority: dict) -> dict:
    ratio = {1: 0.5, 2: 0.3, 3: 0.2}
    counts = {}
    remainder = total

    for priority in [1, 2, 3]:
        chunks = chunks_by_priority.get(priority, [])
        if not chunks:
            continue
        counts[priority] = math.floor(total * ratio[priority])
        remainder -= counts[priority]

    for priority in [1, 2, 3]:
        if priority in counts and remainder > 0:
            counts[priority] += remainder
            remainder = 0
            break

    return counts

# ────────────────────────────────────────
# 프롬프트 생성
# ────────────────────────────────────────
def build_prompt(
    context_text: str,
    keywords: List[str],
    question_type: str,
    feedback_type: Optional[str] = None,
) -> str:
    type_guide = {
        "multiple_choice": """4지선다 객관식 문제.
- 정답은 반드시 1개
- 오답 3개는 그럴싸하지만 명확히 틀린 것으로
- 선지는 비슷한 길이로
- options에 "①","②","③","④" 키로 보기 4개""",

        "ox": """O/X 문제.
- 문장은 명확하게 참 또는 거짓이어야 함
- 애매한 문장 금지
- options는 null""",

        "fill_in_the_blank": """빈칸 채우기 문제.
- 핵심 키워드 자리를 ___로 표시
- 빈칸은 1~2개만
- 정답은 키워드 그대로
- options는 null""",
    }[question_type]

    feedback_guide = ""
    if feedback_type:
        feedback_map = {
            "ambiguous":           "기존 문제가 애매하다는 피드백이 있었어. 더 명확하게 만들어.",
            "wrong_answer":        "정답이 틀렸다는 피드백이 있었어. 정답을 다시 검토해.",
            "unclear_explanation": "해설이 어렵다는 피드백이 있었어. 더 쉽게 써줘.",
            "irrelevant":          "문제가 내용과 관련 없다는 피드백이 있었어. 원문에 더 충실하게 만들어.",
        }
        feedback_guide = f"\n주의: {feedback_map.get(feedback_type, '')}"

    return f"""
너는 대학교 시험 문제 출제 전문가야.
아래 원문을 기반으로 문제를 만들어.

[원문]
{context_text}

[핵심 키워드]
{', '.join(keywords)}

[문제 유형 및 조건]
{type_guide}{feedback_guide}

[주의사항]
- 반드시 원문에 있는 내용만 다뤄
- 원문에 없는 내용 추가 금지
- 해설은 원문 근거를 포함해서 2~3문장으로

반드시 아래 JSON 형식으로만 응답해. 다른 말은 하지 마.
{{
  "question_text": "문제 텍스트",
  "options": {{"①": "...", "②": "...", "③": "...", "④": "..."}},
  "answer": "정답",
  "explanation": "해설"
}}
(OX/빈칸 문제면 options는 null로)
"""

# ────────────────────────────────────────
# Claude 호출
# ────────────────────────────────────────
async def call_claude(prompt: str) -> dict:
    message = await claude_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)

# ────────────────────────────────────────
# GPT 호출
# ────────────────────────────────────────
async def call_gpt(prompt: str) -> dict:
    response = await gpt_client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)

# ────────────────────────────────────────
# 평가 Agent 호출
# ────────────────────────────────────────
async def call_evaluation(
    group_id: str,
    source_chunk_text: str,
    question_text: str,
    options: Optional[dict],
    answer: str,
    explanation: str,
) -> dict:
    payload = {
        "group_id": group_id,
        "source_chunk_text": source_chunk_text,
        "question_text": question_text,
        "options": options,
        "answer": answer,
        "explanation": explanation,
    }
    async with httpx.AsyncClient(timeout=30) as http_client:
        response = await http_client.post(EVALUATION_URL, json=payload)
        response.raise_for_status()
        return response.json()

# ────────────────────────────────────────
# 1. 문제 생성 (document 기준 일괄 생성)
# ────────────────────────────────────────
@router.post("/generate", response_model=QuestionGenerateResponse)
async def generate_questions(
    request: QuestionGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    # 1) document → chunks → importance_results 한번에 가져오기
    stmt = (
        select(ImportanceResult, DocumentChunk, Document)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.id == int(request.group_id))
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        raise HTTPException(status_code=404, detail="해당 문서의 중요도 분석 결과가 없습니다.")

    # 2) User ranking 가져오기
    document = rows[0][2]
    user_q = await db.execute(select(User).where(User.id == document.user_id))
    user = user_q.scalar_one_or_none()

    highlighter_ranking = user.highlighter_ranking or {}
    pen_ranking = user.pen_ranking or {}

    # 3) 청크를 순위별로 분류
    chunks_by_priority = {1: [], 2: [], 3: []}
    for importance, chunk, _ in rows:
        priority = get_priority(
            chunk.meta_data or [],
            highlighter_ranking,
            pen_ranking,
        )
        chunks_by_priority[priority].append((importance, chunk))

    # 4) 순위별 출제 수 계산
    counts = distribute_counts(request.question_count, chunks_by_priority)

    # 5) 문제 생성
    generated: List[QuestionItem] = []

    for priority, target_count in counts.items():
        pool = chunks_by_priority[priority]
        if not pool:
            continue

        pool_sorted = sorted(pool, key=lambda x: x[0].score, reverse=True)

        selected = []
        while len(selected) < target_count:
            selected += pool_sorted
        selected = selected[:target_count]

        for importance, chunk in selected:
            question_type = get_question_type(priority)
            keywords = importance.keywords or []
            prompt = build_prompt(chunk.original_text, keywords, question_type)

            # Claude + GPT 동시 호출
            claude_result, gpt_result = await asyncio.gather(
                call_claude(prompt),
                call_gpt(prompt),
                return_exceptions=True,
            )

            # 평가 Agent에 각각 보내서 점수 받기
            candidates = []
            for source, result in [("claude", claude_result), ("gpt", gpt_result)]:
                if isinstance(result, Exception):
                    continue
                try:
                    review = await call_evaluation(
                        group_id=request.group_id,
                        source_chunk_text=chunk.original_text,
                        question_text=result["question_text"],
                        options=result.get("options"),
                        answer=result["answer"],
                        explanation=result["explanation"],
                    )
                    candidates.append({
                        "source": source,
                        "result": result,
                        "review": review,
                    })
                except Exception:
                    continue

            if not candidates:
                continue

            # 승인된 것 중 quality_score 높은 거 선택
            approved = [c for c in candidates if c["review"]["is_approved"]]

            if approved:
                best = max(approved, key=lambda x: x["review"]["quality_score"])
            else:
                # 둘 다 반려됐지만 suggested_revision 있으면 그걸로 대체
                revised = [c for c in candidates if c["review"].get("suggested_revision_text")]
                if not revised:
                    continue
                best = revised[0]
                best["result"]["question_text"] = best["review"]["suggested_revision_text"]
                if best["review"].get("suggested_revision_options"):
                    best["result"]["options"] = best["review"]["suggested_revision_options"]

            result = best["result"]

            # DB 저장
            question = Question(
                importance_id=importance.id,
                question_type=question_type,
                difficulty=str(priority),
                question_text=result["question_text"],
                options=result.get("options"),
                answer=result["answer"],
                explanation=result["explanation"],
            )
            db.add(question)
            await db.flush()

            generated.append(QuestionItem(
                chunk_id=chunk.id,
                question_type=question_type,
                question_text=result["question_text"],
                options=result.get("options"),
                answer=result["answer"],
                explanation=result["explanation"],
            ))

    await db.commit()
    return QuestionGenerateResponse(questions=generated)


# ────────────────────────────────────────
# 2. 피드백 기반 재생성
# ────────────────────────────────────────
@router.post("/regenerate", response_model=RegenerateResponse)
async def regenerate_question(
    request: RegenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    # 최대 3회 제한
    if request.retry_count >= 3:
        raise HTTPException(status_code=400, detail="재생성은 최대 3회까지만 가능합니다.")

    prompt = build_prompt(
        request.context_text,
        request.keywords,
        request.question_type,
        feedback_type=request.feedback_type,
    )
    result = await call_claude(prompt)

    # 기존 문제 덮어쓰기
    q_result = await db.execute(select(Question).where(Question.id == request.question_id))
    question = q_result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다.")

    question.question_text = result["question_text"]
    question.options       = result.get("options")
    question.answer        = result["answer"]
    question.explanation   = result["explanation"]

    await db.commit()
    await db.refresh(question)

    return RegenerateResponse(
        question_id=question.id,
        question_type=request.question_type,
        **{k: result[k] for k in ["question_text", "options", "answer", "explanation"]},
    )
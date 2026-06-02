import random, json, math, asyncio, httpx, anthropic, openai, os
import uuid as uuid_lib
import re
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from dotenv import load_dotenv

load_dotenv()

from db.models import Question, ImportanceResult, DocumentChunk, Document, User
from agents.question_agent.schemas import (
    DeleteQuizResultRequest,
    DeleteQuizResultResponse,
    QuestionGenerateRequest,
    QuestionGenerateResponse,
    QuestionItem,
    QuestionsByGroupResponse,
    RegenerateRequest,
    RegenerateResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    AnswerResult,
    RegenerateFromWrongRequest,
    QuestionListRequest,
    QuestionListResponse,
    DocumentItem,
    AttemptItem,
)

claude_client = anthropic.AsyncAnthropic()
gpt_client = openai.AsyncOpenAI()

EVALUATION_URL = os.getenv("EVALUATION_URL", "http://127.0.0.1:8000/evaluation/review")

# ────────────────────────────────────────
# 순위 결정
# ────────────────────────────────────────
def get_priority(meta_data: list, highlighter_ranking: dict, pen_ranking: dict) -> int:
    best = 99
    for cue in meta_data or []:
        if not isinstance(cue, dict):
            continue
        color = cue.get("color", "")
        cue_type = cue.get("type", "")
        if cue_type == "highlight":
            rank = highlighter_ranking.get(color, 99)
        elif cue_type == "pen":
            rank = pen_ranking.get(color, 99)
        else:
            continue
        best = min(best, rank)
    return best if best != 99 else 3

# ────────────────────────────────────────
# source_type 결정
# ────────────────────────────────────────
def get_source_type(meta_data: list) -> str:
    for cue in meta_data or []:
        if not isinstance(cue, dict):
            continue
        if cue.get("type") == "pen":
            return "pen"
    return "highlight"

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
- question_text에 "문제:", "보기:" 등의 접두어를 포함하지 마
- question_text는 문제 내용만 순수하게 작성해

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
        temperature=0.7,
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
        temperature=0.7,
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
# 공통 문제 생성 로직
# ────────────────────────────────────────
async def _generate_questions_from_chunks(
    group_id: str,
    chunks_by_priority: dict,
    question_count: int,
    db: AsyncSession,
    quiz_group_id: uuid_lib.UUID = None,
    round_number: int = None,  # ← 문제 생성 회차
) -> List[QuestionItem]:
    counts = distribute_counts(question_count, chunks_by_priority)
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

            claude_result, gpt_result = await asyncio.gather(
                call_claude(prompt),
                call_gpt(prompt),
                return_exceptions=True,
            )

            candidates = []
            for source, result in [("claude", claude_result), ("gpt", gpt_result)]:
                if isinstance(result, Exception):
                    continue
                try:
                    review = await call_evaluation(
                        group_id=group_id,
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

            approved = [c for c in candidates if c["review"]["is_approved"]]
            if approved:
                best = max(approved, key=lambda x: x["review"]["quality_score"])
            else:
                revised = [c for c in candidates if c["review"].get("suggested_revision_text")]
                if not revised:
                    continue
                best = revised[0]
                best["result"]["question_text"] = best["review"]["suggested_revision_text"]
                if best["review"].get("suggested_revision_options"):
                    best["result"]["options"] = best["review"]["suggested_revision_options"]

            result = best["result"]

            question = Question(
                importance_id=importance.id,
                quiz_group_id=quiz_group_id,
                round_number=round_number,  # ← 회차 번호 저장
                question_type=question_type,
                difficulty=str(priority),
                question_text=result["question_text"],
                options=result.get("options"),
                answer=result["answer"],
                explanation=result["explanation"],
            )
            db.add(question)
            await db.flush()

            source_type = get_source_type(chunk.meta_data or [])
            generated.append(QuestionItem(
                chunk_id=chunk.id,
                question_type=question_type,
                question_text=result["question_text"],
                options=result.get("options"),
                answer=result["answer"],
                explanation=result["explanation"],
                question_number=len(generated) + 1,
                priority=priority,
                source_type=source_type,
                page_number=chunk.page_number,
                question_id=question.id,
            ))

    return generated

# ────────────────────────────────────────
# 1. 문제 생성 서비스
# ────────────────────────────────────────
async def generate_questions_service(
    request: QuestionGenerateRequest,
    db: AsyncSession,
) -> QuestionGenerateResponse:

    stmt = (
        select(ImportanceResult, DocumentChunk, Document)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.group_id == request.group_id)
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        raise ValueError("해당 문서의 중요도 분석 결과가 없습니다.")

    document = rows[0][2]
    user_q = await db.execute(select(User).where(User.id == document.user_id))
    user = user_q.scalar_one_or_none()

    highlighter_ranking = user.highlighter_ranking or {}
    pen_ranking = user.pen_ranking or {}

    chunks_by_priority = {1: [], 2: [], 3: []}
    for importance, chunk, _ in rows:
        priority = get_priority(
            chunk.meta_data or [],
            highlighter_ranking,
            pen_ranking,
        )
        chunks_by_priority[priority].append((importance, chunk))

    quiz_group_id = uuid_lib.uuid4()

    # ← 문제 생성 전 현재 document의 max round_number 조회
    sibling_stmt = select(Document.id).where(Document.group_id == document.group_id)
    sibling_ids = [r[0] for r in (await db.execute(sibling_stmt)).all()]

    max_round_stmt = (
        select(func.max(Question.round_number))
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .where(DocumentChunk.document_id.in_(sibling_ids))
    )
    max_round = (await db.execute(max_round_stmt)).scalar() or 0
    next_round = max_round + 1

    generated = await _generate_questions_from_chunks(
        group_id=request.group_id,
        chunks_by_priority=chunks_by_priority,
        question_count=request.question_count,
        db=db,
        quiz_group_id=quiz_group_id,
        round_number=next_round,  # ← 회차 번호 전달
    )

    await db.commit()
    return QuestionGenerateResponse(
        document_id=document.id,
        quiz_group_id=quiz_group_id,
        questions=generated
    )

# ────────────────────────────────────────
# 2. 피드백 기반 재생성 서비스
# ────────────────────────────────────────
async def regenerate_question_service(
    request: RegenerateRequest,
    db: AsyncSession,
) -> RegenerateResponse:

    prompt = build_prompt(
        request.context_text,
        request.keywords,
        request.question_type,
        feedback_type=request.feedback_type,
    )
    result = await call_claude(prompt)

    q_result = await db.execute(select(Question).where(Question.id == request.question_id))
    question = q_result.scalar_one_or_none()

    if not question:
        raise ValueError("문제를 찾을 수 없습니다.")

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

# ────────────────────────────────────────
# 3. 채점 + 저장
# ────────────────────────────────────────
async def submit_answers_service(
    request: SubmitAnswerRequest,
    db: AsyncSession,
) -> SubmitAnswerResponse:
    from db.models import UserAnswer, QuizResult

    results = []

    for item in request.answers:
        question_id = item["question_id"]
        submitted = item["submitted_answer"]

        q_result = await db.execute(
            select(Question).where(Question.id == question_id)
        )
        question = q_result.scalar_one_or_none()
        if not question:
            continue

        # 띄어쓰기 무시하고 채점
        def normalize(text: str) -> str:
            import re
            return re.sub(r'\s+', '', text.strip())

        is_correct = (normalize(submitted) == normalize(question.answer))
        results.append({
            "question_id": question_id,
            "submitted": submitted,
            "correct_answer": question.answer,
            "is_correct": is_correct,
            "explanation": question.explanation,
        })
        
    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    score_percent = round((correct / total) * 100) if total > 0 else 0

    quiz_result = QuizResult(
        user_id=request.user_id,
        document_id=request.document_id,
        quiz_group_id=request.quiz_group_id,
        total_questions=total,
        correct_count=correct,
        score_percent=score_percent,
        attempt_phase=request.attempt_phase,
    )
    db.add(quiz_result)
    await db.flush()

    for r in results:
        answer = UserAnswer(
            quiz_result_id=quiz_result.id,
            question_id=r["question_id"],
            user_answer=r["submitted"],
            is_correct=r["is_correct"],
        )
        db.add(answer)

    await db.commit()

    return SubmitAnswerResponse(
        total=total,
        correct=correct,
        wrong=total - correct,
        results=[
            AnswerResult(
                question_id=r["question_id"],
                submitted_answer=r["submitted"],
                correct_answer=r["correct_answer"],
                is_correct=r["is_correct"],
                explanation=r["explanation"],
            )
            for r in results
        ],
    )

# ────────────────────────────────────────
# 4. 오답 기반 재생성
# ────────────────────────────────────────
async def regenerate_from_wrong_service(
    request: RegenerateFromWrongRequest,
    db: AsyncSession,
) -> QuestionGenerateResponse:
    from db.models import UserAnswer, QuizResult

    wrong_stmt = (
        select(UserAnswer.question_id)
        .join(QuizResult, UserAnswer.quiz_result_id == QuizResult.id)
        .where(QuizResult.user_id == request.user_id)
        .where(QuizResult.document_id == request.document_id)
        .where(UserAnswer.is_correct == False)
    )
    wrong_rows = (await db.execute(wrong_stmt)).all()
    wrong_question_ids = [r[0] for r in wrong_rows]

    wrong_chunk_ids = set()
    if wrong_question_ids:
        stmt = (
            select(DocumentChunk.id)
            .join(ImportanceResult, ImportanceResult.chunk_id == DocumentChunk.id)
            .join(Question, Question.importance_id == ImportanceResult.id)
            .where(Question.id.in_(wrong_question_ids))
        )
        wrong_chunk_rows = (await db.execute(stmt)).all()
        wrong_chunk_ids = {r[0] for r in wrong_chunk_rows}

    stmt = (
        select(ImportanceResult, DocumentChunk, Document)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.group_id == request.group_id)
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        raise ValueError("해당 문서의 중요도 분석 결과가 없습니다.")

    document = rows[0][2]
    user_q = await db.execute(select(User).where(User.id == document.user_id))
    user = user_q.scalar_one_or_none()
    highlighter_ranking = user.highlighter_ranking or {}
    pen_ranking = user.pen_ranking or {}

    chunks_by_priority = {1: [], 2: [], 3: []}
    for importance, chunk, _ in rows:
        if chunk.id in wrong_chunk_ids:
            priority = 1
        else:
            priority = get_priority(
                chunk.meta_data or [],
                highlighter_ranking,
                pen_ranking,
            )
        chunks_by_priority[priority].append((importance, chunk))

    quiz_group_id = uuid_lib.uuid4()

    # ← 문제 재생성 전 현재 document의 max round_number 조회
    sibling_stmt = select(Document.id).where(Document.group_id == document.group_id)
    sibling_ids = [r[0] for r in (await db.execute(sibling_stmt)).all()]

    max_round_stmt = (
        select(func.max(Question.round_number))
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .where(DocumentChunk.document_id.in_(sibling_ids))
    )
    max_round = (await db.execute(max_round_stmt)).scalar() or 0
    next_round = max_round + 1

    generated = await _generate_questions_from_chunks(
        group_id=request.group_id,
        chunks_by_priority=chunks_by_priority,
        question_count=request.question_count,
        db=db,
        quiz_group_id=quiz_group_id,
        round_number=next_round,  # ← 회차 번호 전달
    )

    await db.commit()
    return QuestionGenerateResponse(
        document_id=document.id,
        quiz_group_id=quiz_group_id,
        questions=generated
    )

# ────────────────────────────────────────
# 5. 문서별 생성 문제 리스트 조회
# ────────────────────────────────────────
async def get_question_list_service(
    request: QuestionListRequest,
    db: AsyncSession,
) -> QuestionListResponse:
    from db.models import QuizResult

    doc_stmt = select(Document).where(Document.user_id == request.user_id)
    if request.document_id:
        doc_stmt = doc_stmt.where(Document.id == request.document_id)

    doc_rows = (await db.execute(doc_stmt)).scalars().all()

    if not doc_rows:
        return QuestionListResponse(documents=[])

    # group_id 기준으로 중복 제거
    seen_group_ids = set()
    unique_docs = []
    for doc in doc_rows:
        g = str(doc.group_id)
        if g not in seen_group_ids:
            seen_group_ids.add(g)
            unique_docs.append(doc)

    documents = []
    for doc in unique_docs:
        # 같은 group_id의 document_id 전체 수집
        sibling_stmt = select(Document.id).where(Document.group_id == doc.group_id)
        sibling_ids = [r[0] for r in (await db.execute(sibling_stmt)).all()]

        # quiz_group별로 그룹핑 (sibling document_id 전체 대상)
        group_stmt = (
            select(
                Question.quiz_group_id,
                Question.round_number,
                func.min(Question.created_at).label("created_at"),
                func.count(Question.id).label("q_num")
            )
            .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
            .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
            .where(DocumentChunk.document_id.in_(sibling_ids))
            .where(Question.quiz_group_id.isnot(None))
            .group_by(Question.quiz_group_id, Question.round_number)
            .order_by(Question.round_number.desc())
        )
        group_rows = (await db.execute(group_stmt)).all()

        attempts = []
        for group in group_rows:
            qr_stmt = (
                select(QuizResult)
                .where(QuizResult.document_id.in_(sibling_ids))
                .where(QuizResult.quiz_group_id == group.quiz_group_id)
                .order_by(QuizResult.created_at.desc())
                .limit(1)
            )
            qr = (await db.execute(qr_stmt)).scalar_one_or_none()

            attempts.append(AttemptItem(
                quiz_result_id=qr.id if qr else None,
                quiz_group_id=group.quiz_group_id,
                round=group.round_number if group.round_number else 1,
                created_at=group.created_at.isoformat(),
                q_num=group.q_num,
                score=qr.score_percent if qr and qr.correct_count > 0 else None,
                attempt_phase=qr.attempt_phase if qr else None,
            ))

        documents.append(DocumentItem(
            document_id=doc.id,
            group_id=doc.group_id,
            title=doc.title,
            upload_date=doc.created_at.isoformat() if doc.created_at else "",
            total_count=len(group_rows),
            attempts=attempts,
        ))

    return QuestionListResponse(documents=documents)

# ────────────────────────────────────────
# 6. 회차 삭제
# ────────────────────────────────────────
async def delete_quiz_results_service(
    request: DeleteQuizResultRequest,
    db: AsyncSession,
) -> DeleteQuizResultResponse:
    from db.models import UserAnswer, QuizResult
    from uuid import UUID as UUIDType

    deleted_count = 0

    # quiz_group_id 기준으로 questions + quiz_results + user_answers 삭제
    for group_id_str in request.quiz_group_ids:
        group_id = UUIDType(group_id_str)

        # 해당 그룹의 quiz_results 조회
        qr_stmt = select(QuizResult).where(QuizResult.quiz_group_id == group_id)
        quiz_results = (await db.execute(qr_stmt)).scalars().all()

        # 본인 데이터인지 검증
        for qr in quiz_results:
            if qr.user_id != request.user_id:
                raise PermissionError("삭제 권한이 없습니다.")
            await db.delete(qr)

        # 해당 그룹의 questions 삭제
        q_stmt = select(Question).where(Question.quiz_group_id == group_id)
        questions = (await db.execute(q_stmt)).scalars().all()
        for q in questions:
            await db.delete(q)

        deleted_count += 1

    await db.commit()

    return DeleteQuizResultResponse(
        deleted_count=deleted_count,
        message="성공적으로 삭제되었습니다."
    )

# ────────────────────────────────────────
# 7. quiz_group_id로 문제 조회
# ────────────────────────────────────────
async def get_questions_by_group_service(
    quiz_group_id: str,
    db: AsyncSession,
) -> QuestionsByGroupResponse:
    from uuid import UUID as UUIDType

    stmt = (
        select(Question, ImportanceResult, DocumentChunk)
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .where(Question.quiz_group_id == UUIDType(quiz_group_id))
        .order_by(Question.id.asc())
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        raise ValueError("해당 문제 그룹을 찾을 수 없습니다.")

    questions = []
    for idx, (question, importance, chunk) in enumerate(rows):
        source_type = get_source_type(chunk.meta_data or [])
        priority = int(question.difficulty) if question.difficulty else 3
        options = question.options if question.options else None

        questions.append(QuestionItem(
            chunk_id=chunk.id,
            question_id=question.id,
            question_type=question.question_type,
            question_text=question.question_text,
            options=options,
            answer=question.answer,
            explanation=question.explanation,
            question_number=idx + 1,
            priority=priority,
            source_type=source_type,
            page_number=chunk.page_number,
        ))

    return QuestionsByGroupResponse(
        quiz_group_id=UUIDType(quiz_group_id),
        questions=questions,
    )

# ────────────────────────────────────────
# 8. 오답 조회
# ────────────────────────────────────────
async def get_wrong_answers_service(
    quiz_result_id: int,
    db: AsyncSession,
):
    from db.models import UserAnswer
    from agents.question_agent.schemas import WrongAnswerItem, WrongAnswersResponse

    stmt = (
        select(UserAnswer, Question, ImportanceResult, DocumentChunk)
        .join(Question, UserAnswer.question_id == Question.id)
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .where(UserAnswer.quiz_result_id == quiz_result_id)
        .where(UserAnswer.is_correct == False)
    )
    rows = (await db.execute(stmt)).all()

    wrong_answers = []
    for user_answer, question, importance, chunk in rows:
        priority = int(question.difficulty) if question.difficulty else 3

        # ← quiz_group 내 question_number 계산
        order_stmt = (
            select(func.count(Question.id))
            .where(Question.quiz_group_id == question.quiz_group_id)
            .where(Question.id <= question.id)
        )
        question_number = (await db.execute(order_stmt)).scalar() or 0

        wrong_answers.append(WrongAnswerItem(
            question_id=question.id,
            question_number=question_number,  # ← 추가
            question_type=question.question_type,
            question_text=question.question_text,
            options=question.options,
            answer=question.answer,
            explanation=question.explanation,
            submitted_answer=user_answer.user_answer or "",
            page_number=chunk.page_number,
            priority=priority,
        ))

    return WrongAnswersResponse(
        quiz_result_id=quiz_result_id,
        wrong_answers=wrong_answers,
    )

# ────────────────────────────────────────
# 9. 채점 결과 상세 조회
# ────────────────────────────────────────
async def get_quiz_result_detail_service(
    quiz_result_id: int,
    db: AsyncSession,
):
    from db.models import UserAnswer, QuizResult
    from agents.question_agent.schemas import QuizResultDetailResponse

    qr_stmt = select(QuizResult).where(QuizResult.id == quiz_result_id)
    qr = (await db.execute(qr_stmt)).scalar_one_or_none()
    if not qr:
        raise ValueError("채점 결과를 찾을 수 없습니다.")

    ua_stmt = (
        select(UserAnswer, Question)
        .join(Question, UserAnswer.question_id == Question.id)
        .where(UserAnswer.quiz_result_id == quiz_result_id)
    )
    rows = (await db.execute(ua_stmt)).all()

    results = []
    for user_answer, question in rows:
        results.append(AnswerResult(
            question_id=question.id,
            submitted_answer=user_answer.user_answer or "",
            correct_answer=question.answer,
            is_correct=user_answer.is_correct,
            explanation=question.explanation,
        ))

    return QuizResultDetailResponse(
        total=qr.total_questions,
        correct=qr.correct_count,
        wrong=qr.total_questions - qr.correct_count,
        results=results,
    )
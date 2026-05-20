import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.models import Question, Document, QuizResult, UserAnswer, ImportanceResult, DocumentChunk
from agents.personalized_agent.schemas import PersonalizationResponse

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# ────────────────────────────────────────
# 1. 풀이 세션 전체 기록 및 스코어 가중 드라이빙
# ────────────────────────────────────────
async def record_quiz_session(
    user_id: int,
    group_id: str,
    total_questions: int,
    correct_count: int,
    score_percent: int,
    attempt_phase: str,
    answers_list: list,
    db: AsyncSession
):
    doc_result = await db.execute(
        select(Document.id).where(Document.group_id == group_id).limit(1)
    )
    document_id = doc_result.scalar()
    if not document_id:
        raise ValueError(f"group_id '{group_id}'에 해당하는 문서를 찾을 수 없습니다.")

    # 1-1. 기획팀 양식 테이블에 완벽히 매칭
    quiz_master = QuizResult(
        user_id=user_id,
        document_id=document_id,
        total_questions=total_questions,
        correct_count=correct_count,
        score_percent=score_percent,
        attempt_phase=attempt_phase
    )
    db.add(quiz_master)
    await db.flush() 

    # 1-2. 낱개 문제 정오답 기록 및 출제 정렬 스코어 조작
    for ans in answers_list:
        user_ans = UserAnswer(
            quiz_result_id=quiz_master.id, 
            question_id=ans["question_id"],
            user_answer=ans["user_answer"],
            is_correct=ans["is_correct"]
        )
        db.add(user_ans)

        # Question Agent가 소팅하는 중요도 테이블 조회
        stmt = (
            select(ImportanceResult)
            .join(Question, Question.importance_id == ImportanceResult.id)
            .where(Question.id == ans["question_id"])
        )
        imp_res = (await db.execute(stmt)).scalar_one_or_none()
        
        if imp_res:
            if attempt_phase != "first_attempt":
                score_modifier = -5.0 if ans["is_correct"] else 7.5
            else:
                score_modifier = -2.5 if ans["is_correct"] else 10.0
            
            # base_score 필드 하한선 방어
            base_val = getattr(imp_res, "base_score", 0.0)
            imp_res.score = max(base_val, imp_res.score + score_modifier)

    await db.commit()
    return True


# ────────────────────────────────────────
# 2. 7:3 통합을 위한 group_id 기반 취약점 팩트 체크 분석
# ────────────────────────────────────────
async def analyze_weakness(user_id: int, group_id: str, db: AsyncSession) -> PersonalizationResponse:
    # 💡 다른 팀원들의 핵심 연결 고리(group_id ➡️ Document ➡️ Chunk)를 모두 활용한 완벽한 쿼리
    stmt = (
        select(
            Question.question_text, 
            ImportanceResult.keywords, 
            UserAnswer.is_correct
        )
        .join(UserAnswer, Question.id == UserAnswer.question_id)
        .join(QuizResult, UserAnswer.quiz_result_id == QuizResult.id)
        .join(ImportanceResult, Question.importance_id == ImportanceResult.id)
        .join(DocumentChunk, ImportanceResult.chunk_id == DocumentChunk.id)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(QuizResult.user_id == user_id, Document.group_id == group_id)
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        return PersonalizationResponse(
            weakness_concepts=["아직 그룹 내 풀이 이력이 존재하지 않습니다."],
            weakness_source_file=group_id,
            personalized_advice="문제를 먼저 풀어보시면 정밀 약점 진단 리포트가 생성됩니다.",
            next_recommendation=[]
        )

    # 오답 분석 팩트 트리밍
    wrong_questions_details = []
    total_count = 0
    wrong_count = 0
    
    for row in rows:
        total_count += 1
        if not row.is_correct:
            wrong_count += 1
            k_list = row.keywords if isinstance(row.keywords, list) else json.loads(row.keywords or "[]")
            wrong_questions_details.append(
                f"- [틀린 문제]: {row.question_text}\n  [개념 키워드]: {', '.join(k_list)}"
            )

    stats_summary = f"그룹 전체 {total_count}문제 중 {wrong_count}문제를 오답 처리했습니다.\n" + "\n".join(wrong_questions_details[:10])

    # AI 조언 추출 프롬프트
    prompt = f"""
학생이 학습 중인 그룹 코드: {group_id}
학생의 누적 오답 팩트 데이터:
{stats_summary}

제공된 실제 오답 리스트에만 철저히 근거하여 학생의 논리적 약점을 요약하세요.
다른 부연 설명 없이 아래 JSON 포맷으로만 응답하세요.

{{
  "weakness_concepts": ["반복 오답 기반 핵심 취약 개념 1", "개념 2"],
  "personalized_advice": "놓치고 있는 이론적 맹점 분석 및 올바른 복습 처방전 (1문단)",
  "next_recommendation": ["다음 문제 생성 시 우선 마스터해야 할 키워드 1", "키워드 2"]
}}
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    clean_content = response.content.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(clean_content)

    return PersonalizationResponse(
        weakness_concepts=parsed["weakness_concepts"],
        weakness_source_file=f"학습 그룹: {group_id}",
        personalized_advice=parsed["personalized_advice"],
        next_recommendation=parsed["next_recommendation"]
    )
from dotenv import load_dotenv
load_dotenv()  

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager

from db.database import engine, Base
import db.models as models 

from agents.importance_agent.router import router as importance_router
from agents.evaluation_agent.router import router as evaluation_router
from agents.question_agent.router import router as question_router
from agents.retrieval_agent.router import router as retrieval_router
from agents.personalized_agent.router import router as personalized_router
from api.workflow import router as workflow_router

from api.users import router as users_router

# 서버가 켜질 때 자동으로 실행될 준비(시작) 작업을 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Supabase DB 테이블 생성 완료!")
    
    yield 

# FastAPI 앱 객체 생성
app = FastAPI(
    title="Highlite 멀티 에이전트 API",
    description="교재 PDF 중요도 분석 및 맞춤형 문제 생성 AI 서버",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 미들웨어 설정 (프론트엔드 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 -> 실제 배포 시에는 보안을 위해 특정 도메인만 허용하도록 변경 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("🚨 .env 파일에 SESSION_SECRET_KEY가 설정되지 않았습니다!")

# 구글 로그인을 위한 세션 미들웨어 
app.add_middleware(
    SessionMiddleware, 
    secret_key=SECRET_KEY 
)

@app.get("/")
def read_root():
    return {"message": "Highlite 멀티 에이전트 서버가 정상 작동 중입니다!"}

app.include_router(users_router)

app.include_router(
    importance_router, 
    prefix="/importance",  
    tags=["중요도 분석 Agent"]
)

app.include_router(
    evaluation_router, 
    prefix="/evaluation",  
    tags=["평가 Agent"]
)

app.include_router(
    personalized_router, 
    prefix="/personalized",  
    tags=["개인화 Agent"]
)

app.include_router(
    question_router, 
    prefix="/question",  
    tags=["문제 생성 Agent"]
)

app.include_router(
    retrieval_router, 
    prefix="/retrieval",  
    tags=["RAG 검색 Agent"]
)

app.include_router(
    workflow_router, 
    prefix="/api/v1", 
    tags=["Production"]
)
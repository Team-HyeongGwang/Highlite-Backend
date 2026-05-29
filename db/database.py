# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(
    DATABASE_URL, 
    echo=True,
    connect_args={"statement_cache_size": 0},
    pool_pre_ping=True,  # ← 추가 (쿼리 전 커넥션 살아있는지 확인)
    pool_recycle=3600,   # ← 추가 (1시간마다 커넥션 재생성)  
)

AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
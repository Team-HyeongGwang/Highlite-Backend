import os
import jwt
import random
from pydantic import BaseModel
from datetime import datetime, timedelta
from dotenv import load_dotenv

from fastapi import APIRouter, Request, Depends
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete

from db.database import get_db 
import db.models as models

load_dotenv()  

router = APIRouter(prefix="/users", tags=["Users"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "highlite_super_secret_key")

oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

class NicknameUpdate(BaseModel):
    email: str
    new_nickname: str

class AccountDelete(BaseModel):
    email: str

# JWT 토큰 생성 함수
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24) # 토큰 유효기간 24시간
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm="HS256")
    return encoded_jwt

# 구글 로그인 창으로 보내는 API
@router.get("/login/google")
async def login_via_google(request: Request):
    redirect_uri = "http://localhost:8000/users/login/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri, prompt="select_account")

def generate_random_nickname():
    adjectives = ["열공하는", "졸린", "행복한", "코딩하는", "똑똑한", "신비로운", "빛나는", "용감한", "귀여운", "멋진", "재미있는", "창의적인", "친절한", "유쾌한", "상냥한", "똑부러지는", "센스있는", "사려깊은", "낙천적인", "열정적인", "차분한", "긍정적인", "낭만적인", "섬세한", "지혜로운", "활발한", "사랑스러운", "유머러스한"]
    nouns = ["알파카", "거북이", "올빼미", "고양이", "강아지", "펭귄", "호랑이", "여우", "사자", "코끼리", "원숭이", "판다", "토끼", "용", "유니콘", "드래곤", "고릴라", "코알라", "늑대", "독수리", "돌고래", "고래", "사슴", "부엉이", "코뿔소", "낙타", "하마", "악어", "카멜레온", "코브라", "스컹크", "두더지", "라쿤", "오소리", "족제비", "담비", "수달", "바다표범", "물개", "펠리컨", "플라밍고", "앵무새", "까마귀", "비둘기", "참새", "제비", "갈매기"]
    random_num = random.randint(1000, 9999)
    return f"{random.choice(adjectives)}_{random.choice(nouns)}_{random_num}"

@router.get("/login/google/callback")
async def auth_google_callback(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        email = user_info.get("email")
        google_picture_url = user_info.get("picture") 
        
        query = select(models.User).where(models.User.email == email)
        result = await db.execute(query)
        user = result.scalars().first()
        
        # 처음 온 유저라면 DB에 새로 저장
        if not user:
            user = models.User(
                email=email,
                username=generate_random_nickname(),
                provider="google",
                profile_image_url=google_picture_url 
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        
        if user.join_date:
            join_date_str = user.join_date.strftime("%Y.%m.%d")
        else:
            from datetime import datetime
            join_date_str = datetime.utcnow().strftime("%Y.%m.%d")

        token_data = {
            "sub": user.email, 
            "user_id": user.id,
            "username": user.username,
            "picture": user.profile_image_url,
            "join_date": join_date_str 
        }
        highlite_access_token = create_access_token(token_data)
        
        FRONTEND_URL = f"http://localhost:8501/?token={highlite_access_token}"
        return RedirectResponse(url=FRONTEND_URL)

    except Exception as e:
        print(f"🔥 소셜 로그인 중 에러 발생: {str(e)}")
        await db.rollback() 
        return RedirectResponse(url="http://localhost:8501/?error=login_failed")

@router.put("/nickname")
async def update_nickname(data: NicknameUpdate, db: AsyncSession = Depends(get_db)):
    query = select(models.User).where(models.User.email == data.email)
    result = await db.execute(query)
    user = result.scalars().first()
    
    if user:
        user.username = data.new_nickname
        await db.commit()
        return {"status": "success", "new_nickname": user.username}
    
    return {"status": "error", "message": "유저를 찾을 수 없습니다."}

@router.delete("/account")
async def delete_account(data: AccountDelete, db: AsyncSession = Depends(get_db)):
    try:
        # 이메일로 유저를 찾아서 삭제
        query = delete(models.User).where(models.User.email == data.email)
        await db.execute(query)
        await db.commit()
        return {"status": "success", "message": "회원 탈퇴가 완료되었습니다."}
    except Exception as e:
        await db.rollback()
        return {"status": "error", "message": "탈퇴 처리 중 오류가 발생했습니다."}
# 🧠 Highlite-Backend
> [Highlite 프로젝트 설명 바로 가기](https://github.com/Team-HyeongGwang)

</br>

## ✨ Main 기능
- PDF 업로드 및 OCR 파이프라인
- 형광펜·필기펜 중요도 분석
- 시험 문제 자동 생성 및 검수
- 오답 기반 개인화 재출제
- 요약본 생성 및 내보내기

</br>

## 👩‍💻 역할 분담

| 이름 | 역할 |
|---|---|
| 임지영 | RAG Agent |
| 송유진 | 중요도 분석 Agent |
| 김채현 | 문제 생성 Agent |
| 김서형 | 평가 Agent, 개인화 Agent |

</br>

## 🌳 프로젝트 구조
```
Highlite/
├── main.py
├── requirements.txt
├── .gitignore
├── .github/
├── agents/
│   ├── evaluation_agent/
│   │   ├── router.py
│   │   ├── schemas.py
│   │   └── service.py
│   ├── importance_agent/
│   │   ├── router.py
│   │   ├── schemas.py
│   │   └── service.py
│   ├── personalized_agent/
│   │   ├── router.py
│   │   ├── schemas.py
│   │   └── service.py
│   ├── question_agent/
│   │   ├── router.py
│   │   ├── schemas.py
│   │   └── service.py
│   └── retrieval_agent/
│       ├── primer.py
│       ├── router.py
│       ├── schemas.py
│       ├── service.py
│       └── test_run.py
├── api/
│   ├── export.py
│   ├── users.py
│   └── workflow.py
├── common/
│   └── schemas.py
├── db/
│   ├── database.py
│   ├── models.py
│   └── supabase_client.py
├── ranks/
│   ├── router.py
│   ├── schemas.py
│   └── service.py
└── utils/
```

</br>

## 🚀 실행 방법
```bash
uvicorn main:app --reload
```

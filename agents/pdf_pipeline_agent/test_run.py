from pathlib import Path
from pdf_pipeline_agent import extract_from_pdf
from dotenv import load_dotenv

load_dotenv()

def run_test():
    test_pdf = Path("pdf 경로 추가")

    if not test_pdf.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {test_pdf.absolute()}")
        return

    print(f"🚀 테스트 시작: {test_pdf.name}")
    print("-" * 50)

    try:
        results = extract_from_pdf(test_pdf)

        print(f"✅ 총 {len(results)}개의 문단이 추출되었습니다.")

        if len(results) == 0:
            print("⚠️ 텍스트가 추출되지 않았습니다.")
            return

        print("-" * 50)
        for i, res in enumerate(results[:5]):
            print(f"--- [샘플 {i+1}] Page: {res['page']}, Para: {res['paragraph_index']} ---")
            print(f"내용: {res['content'][:150]}")

            content = res['content']
            for tag, label in [("[손필기:", "✍️ 손필기"), ("[형광펜:", "🖊️ 형광펜"), ("[주석:", "📌 주석")]:
                if tag in content:
                    print(f"  {label} 감지됨")

            print()

    except Exception as e:
        print(f"🚨 테스트 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    run_test()
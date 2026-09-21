"""README용 데모 화면 캡처. streamlit run app.py 가 localhost:8501에서 떠 있어야 한다."""
import time
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(URL)
    page.wait_for_selector("input[type=text]", timeout=15000)
    time.sleep(1)

    # 1) 엔비디아 질문 -> 답변 + 경로 + 근거
    page.fill("input[type=text]", "엔비디아는 누가 설립했나요?")
    page.keyboard.press("Tab")
    time.sleep(1)
    page.click("button:has-text('질문하기')")
    time.sleep(2)
    page.screenshot(path="docs/_debug_after_click.png", full_page=True)
    page.wait_for_selector("text=탄 경로", timeout=60000)
    time.sleep(1)
    page.screenshot(path="docs/screenshot_answer.png", full_page=True)
    print("saved docs/screenshot_answer.png")

    # 2) 삼성전자 질문 -> 거부 사례
    page.fill("input[type=text]", "삼성전자가 투자한 AI 스타트업은 어디인가요?")
    page.keyboard.press("Tab")
    time.sleep(1)
    page.click("button:has-text('질문하기')")
    time.sleep(2)
    page.wait_for_selector("text=거부 사유", timeout=60000)
    time.sleep(1)
    page.screenshot(path="docs/screenshot_refuse.png", full_page=True)
    print("saved docs/screenshot_refuse.png")

    browser.close()

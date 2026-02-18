import asyncio
from playwright.async_api import async_playwright
from utils import *
from steps import build_scenario

HEADLESS = False #True

TOTAL_USERS = 1 #30                       # 총 “사용자(세션)” 수
CONCURRENCY = 1 #5                        # 동시에 실행할 수


USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

async def simulate_one_user(pw, idx: int):
    browser = await pw.chromium.launch(headless=HEADLESS)

    context = await browser.new_context(
        user_agent=random.choice(USER_AGENTS),
        viewport={"width": 1280, "height": 900},
        locale="ko-KR",
    )

    page = await context.new_page()

    entry = build_utm_url()
    await page.goto(entry, wait_until="domcontentloaded")
    await page.wait_for_timeout(random.randint(800, 1800))

    scenario = build_scenario()
    for step in scenario:
        await step.action(page)
        await dwell(page, *step.dwell_range)

    await context.close()
    await browser.close()

    print(f"user#{idx} done | entry={entry}")

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    async with async_playwright() as pw:

        async def runner(i):
            async with sem:
                await simulate_one_user(pw, i)

        await asyncio.gather(*(runner(i) for i in range(TOTAL_USERS)))

if __name__ == "__main__":
    asyncio.run(main())
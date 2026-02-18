import random, urllib.parse
from playwright.async_api import Page

BASE_URL = "https://giyonglee.com"

SOURCES = ["gmail", "kakao"]
MEDIUMS = ["link", "email"]
CAMPAIGNS = ["portfolio_share", "2026_recruit"]
CONTENTS = ["a", "b", "c", "d"]

SCROLL_CONTAINER = "#main-content"

def build_utm_url():
    utm_source = random.choice(SOURCES)
    utm_medium = random.choice(MEDIUMS)
    utm_campaign = random.choice(CAMPAIGNS)
    params = {
        "utm_source": utm_source,
        "utm_medium": utm_medium,
        "utm_campaign": utm_campaign,
    }
    if utm_campaign == "2026_recruit":
        params["utm_content"] = random.choice(CONTENTS)

    return BASE_URL + "?" + urllib.parse.urlencode(params)

def rand_ms(a: int, b: int) -> int:
    return random.randint(a, b)

async def scroll_container(page: Page, selector: str = SCROLL_CONTAINER, dy: int = 800) -> bool:
    """
    window가 아니라 'overflow 스크롤 컨테이너'를 스크롤.
    성공하면 True.
    """
    try:
        await page.wait_for_selector(selector, timeout=3000)
        await page.evaluate(
            """({sel, dy}) => {
                const el = document.querySelector(sel);
                if (!el) return false;
                el.scrollBy(0, dy);
                return true;
            }""",
            {"sel": selector, "dy": dy},
        )
        return True
    except:
        return False

async def dwell(page: Page, min_ms: int, max_ms: int):
    total = random.randint(min_ms, max_ms)
    elapsed = 0
    while elapsed < total:
        # 70% 확률로 스크롤 시도
        if random.random() < 0.7:
            dy = random.randint(250, 1100)
            # 컨테이너 스크롤 우선
            ok = await scroll_container(page, dy=dy)
            # fallback: 그래도 안 되면 wheel
            if not ok:
                try:
                    await page.mouse.wheel(0, dy)
                except:
                    pass
        await page.wait_for_timeout(random.randint(600, 1600))
        elapsed += 1200

async def wait_spa_settle(page: Page):
    """Vue SPA에서 클릭 후 네트워크/렌더링 안정화용"""
    try:
        await page.wait_for_load_state("networkidle", timeout=5000)
    except:
        pass
    await page.wait_for_timeout(random.randint(300, 900))

async def click_random_match(page: Page, selector: str, timeout_ms: int = 7000) -> bool:
    """
    selector에 매칭되는 요소들 중 랜덤 1개 클릭.
    - '포스트 카드 여러 개 중 하나' 같은 케이스를 위해 필요.
    """
    try:
        await page.wait_for_selector(selector, timeout=timeout_ms)
    except:
        return False

    els = await page.query_selector_all(selector)
    if not els:
        return False

    el = random.choice(els)
    try:
        await el.click()
        return True
    except:
        return False

async def is_post_detail(page: Page) -> bool:
    """현재 URL이 /posts/{id} 형태인지 대략 판단."""
    return "/posts/" in page.url.rstrip("/")
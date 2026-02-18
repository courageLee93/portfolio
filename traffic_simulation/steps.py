from utils import *
from dataclasses import dataclass
from typing import Callable, Awaitable, List
from playwright.async_api import Page
import random

@dataclass
class Step:
    name: str
    action: Callable[[Page], Awaitable[None]]
    # step이 끝난 후 체류시간(페이지별) 범위
    dwell_range: tuple[int, int] = (1200, 3500)

# 1) 홈: 포스트(/posts/:id)로 바로 들어가야 함
async def step_home_to_post(page: Page):
    await wait_spa_settle(page)
    # 홈에서 짧게 체류
    await dwell(page, 1500, 4500)

    # DOM 기반: a[href="/posts/1"] + data-post-id
    # - 여러 포스트 카드 중 랜덤 1개 클릭
    ok = await click_random_match(page, 'a[href^="/posts/"][data-post-id]')
    if ok:
        await wait_spa_settle(page)

# 2) 상세: 스크롤/체류
async def step_post_detail_scroll(page: Page):
    await wait_spa_settle(page)

    # 상세 도착이 아닐 수도 있으니 체크 (안 맞으면 스킵)
    if not await is_post_detail(page):
        return

    # 상세에서 길게 체류 + 스크롤 (GA4 engagement_time 증가)
    await dwell(page, 8000, 20000)

    # 추가 스크롤(깊은 스크롤)
    for _ in range(random.randint(3, 8)):
        dy = random.randint(300, 1400)
        ok = await scroll_container(page, dy=dy)
        if not ok:
            await page.mouse.wheel(0, dy)
        await page.wait_for_timeout(random.randint(400, 1200))

# 3) 상세: outbound 클릭(논문/리포트/코드)
async def step_post_detail_outbound(page: Page):
    await wait_spa_settle(page)

    if not await is_post_detail(page):
        return

    if random.random() >= 0.75:
        return

    ctx = page.context
    before_pages = ctx.pages.copy()

    # 클릭
    ok = await click_random_match(page, ".gtm-outbound-click")
    if not ok:
        return

    await page.wait_for_timeout(1500)  # 클릭 이벤트 전송 시간

    after_pages = ctx.pages

    # 새 탭이 생겼는지 확인
    if len(after_pages) > len(before_pages):
        new_page = [p for p in after_pages if p not in before_pages][0]

        try:
            await new_page.wait_for_load_state("domcontentloaded", timeout=5000)
        except:
            pass

        # 실제로 잠깐 보이게 유지
        await new_page.wait_for_timeout(random.randint(2000, 4000))

        await new_page.close()

        # 원래 탭으로 포커스
        await page.bring_to_front()

    else:
        # 같은 탭에서 열렸다면 → 다시 돌아오기
        if not page.url.startswith(BASE_URL):
            await page.wait_for_timeout(2000)
            await page.goto(BASE_URL, wait_until="domcontentloaded")
            await wait_spa_settle(page)

# 4) 홈으로 돌아가기: logo 클릭
async def step_post_to_home(page: Page) -> bool:
    """
    .logo 클릭으로 홈 복귀.
    실패 시 직접 BASE_URL로 이동.
    """
    try:
        await page.wait_for_selector(".logo", timeout=5000)
        await page.click(".logo")
        await wait_spa_settle(page)
        return True
    except:
        # 혹시 실패하면 직접 이동
        try:
            await page.goto(BASE_URL, wait_until="domcontentloaded")
            await wait_spa_settle(page)
            return True
        except:
            return False

# 5) 홈으로 돌아간 뒤, 다른 포스트들을 몇 번 더 반복 탐색
async def step_loop_more_posts(page: Page):
    """
    홈 -> 포스트 -> 상세 스크롤 -> outbound -> 홈
    을 랜덤 횟수로 반복해서 세션 길이/페이지 수를 다양화.
    """
    loops = random.randint(0, 5)  # 추가 탐색 횟수

    for _ in range(loops):
        # (a) 홈으로 복귀
        await step_post_to_home(page)
        # (b) 홈에서 조금 체류(자연스럽게)
        await dwell(page, 1200, 4000)
        # (c) 다른 포스트 클릭
        ok = await click_random_match(page, 'a[href^="/posts/"][data-post-id]')
        if not ok:
            return
        await wait_spa_settle(page)
        # (d) 상세에서 스크롤/체류
        await step_post_detail_scroll(page)
        # (e) outbound 클릭(확률)
        await step_post_detail_outbound(page)

# -------------------------
# 시나리오 구성
# -------------------------
def build_scenario() -> List[Step]:
    """
    entry(홈) -> 포스트 클릭 -> 상세 스크롤/체류 -> outbound 클릭
    -> (loop) 홈으로 복귀 후 다른 포스트 반복
    """
    steps: List[Step] = [
        Step("home_to_post", step_home_to_post, (500, 1500)),
        Step("post_detail_scroll", step_post_detail_scroll, (500, 1500)),
        Step("post_detail_outbound", step_post_detail_outbound, (500, 1500)),
        Step("loop_more_posts", step_loop_more_posts, (500, 1500)),
    ]
    return steps


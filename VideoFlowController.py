from __future__ import annotations

import time
from typing import Optional

from playwright.sync_api import Page, Locator

from human_cursor import HumanCursor, connect_cdp, first_existing_page
from engagement_simulator2 import EngagementSimulator, EngagementProfile


VIDEO_DURATION_JS = """
() => {
    const video = document.querySelector('video');
    if (!video) return null;

    const duration = Number(video.duration);
    return Number.isFinite(duration) && duration > 0 ? duration : null;
}
"""


class VideoFlowController:
    """
    QA/staging-контроллер.

    Делает:
      1. подключается к уже открытому браузеру через CDP;
      2. получает duration текущего <video>;
      3. рассчитывает dwell через EngagementSimulator;
      4. выдерживает dwell;
      5. прокручивает страницу к следующему элементу.

    Намеренный engagement-клик отсутствует.
    """

    def __init__(
        self,
        page: Page,
        simulator: EngagementSimulator,
        cursor: Optional[HumanCursor] = None,
    ) -> None:
        self.page = page
        self.simulator = simulator
        self.cursor = cursor
        self.session_index = 0

    def get_video_duration(self) -> float:
        duration = self.page.evaluate(VIDEO_DURATION_JS)

        if duration is None:
            raise RuntimeError("На странице не найдено активное видео с известной duration")

        return float(duration)

    def sample_dwell(self, duration: float) -> float:
        return self.simulator.sample(
            video_duration_sec=duration,
            session_index=self.session_index,
        )

    def wait_dwell(self, dwell: float) -> None:
        if dwell <= 0:
            return

        print(
            f"[video #{self.session_index}] "
            f"dwell={dwell:.2f}s"
        )
        time.sleep(dwell)

    def scroll_to_next(self, locator: Locator) -> None:
        """
        Прокрутка к следующему элементу.

        Для QA можно использовать существующий HumanCursor,
        но только как средство перемещения в тестовой среде.
        """
        locator.wait_for(state="visible")

        if self.cursor is not None:
            self.cursor.move_to_locator(locator)
        else:
            locator.scroll_into_view_if_needed()

    def run_once(self, next_locator: Locator) -> dict:
        duration = self.get_video_duration()
        dwell = self.sample_dwell(duration)

        self.wait_dwell(dwell)
        self.scroll_to_next(next_locator)

        result = {
            "session_index": self.session_index,
            "video_duration": duration,
            "dwell": dwell,
        }

        self.session_index += 1
        return result


def run_qa_flow(
    cdp_endpoint: str,
    next_selector: str,
    profile: EngagementProfile,
    seed: int | None = None,
) -> None:
    simulator = EngagementSimulator(profile, seed=seed)

    with connect_cdp(cdp_endpoint) as browser:
        page = first_existing_page(browser)

        cursor = HumanCursor(page)
        controller = VideoFlowController(
            page=page,
            simulator=simulator,
            cursor=cursor,
        )

        next_locator = page.locator(next_selector)

        result = controller.run_once(next_locator)
        print(result)


if __name__ == "__main__":
    run_qa_flow(
        cdp_endpoint="ws://localhost:4444/devtools/...",
        next_selector="[data-testid='next-video']",
        profile=EngagementProfile.from_mode_median(
            "qa",
            mode_sec=3.0,
            median_sec=10.0,
        ),
        seed=42,
    )
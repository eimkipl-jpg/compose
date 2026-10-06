"""
human_cursor.py
================

Обёртка над Playwright для генерации человекоподобных траекторий курсора.

Назначение: калибровочный харнесс / генератор поведенческих фикстур.
Движение строится по кубическим кривым Безье с рандомизированными
контрольными точками, добавляются микро-дрожание (jitter), перелёт
(overshoot) и неравномерное распределение точек по времени (ease-in-out),
чтобы получить реалистичную вариативность "человеческих" сэмплов.

Физика движения вынесена в профили (MovementProfile). Рандомизация
контрольных точек и амплитуд даёт разные дуги на каждом вызове — это
свойство реалистичной модели движения, а не настройка против какого-либо
детектора.

Движения отправляются через page.mouse.move(...), поэтому промежуточные
hover-координаты реально проходят через браузер и фиксируются аналитикой
(в отличие от locator.click(), который "телепортирует" курсор).
"""

from __future__ import annotations

import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

from playwright.sync_api import Browser, Locator, Page, sync_playwright

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
#  Профили движения
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MovementProfile:
    """Параметры физики движения.

    Все величины, влияющие на форму траектории, скорость и тайминги,
    вынесены сюда. Меняя профиль, получаем разный "характер" пользователя.
    """

    name: str

    # --- Плотность точек / "скорость" -------------------------------------- #
    # Сколько промежуточных точек генерировать на каждые 100 px пути.
    # Больше точек -> плавнее и "медленнее" (меньше пиксельный шаг).
    steps_per_100px: float
    min_steps: int
    max_steps: int

    # --- Форма кривой ------------------------------------------------------ #
    # Базовая амплитуда отклонения контрольных точек (доля от длины отрезка).
    curve_amplitude: float
    # Случайный разброс амплитуды (±), чтобы дуги не повторялись.
    curve_amplitude_jitter: float

    # --- Микро-дрожание ---------------------------------------------------- #
    jitter_px: float  # сигма гауссова шума на промежуточных точках

    # --- Перелёт ----------------------------------------------------------- #
    overshoot_probability: float
    overshoot_px: Tuple[float, float]  # диапазон величины перелёта

    # --- Тайминги ---------------------------------------------------------- #
    step_delay: Tuple[float, float]       # пауза между микродвижениями, сек
    pre_click_pause: Tuple[float, float]  # пауза "наведение -> клик", сек

    # --- Распределение точек по времени ------------------------------------ #
    # Сила ease-in-out: 1.0 = равномерно (линейно), >1 = плавный разгон/замедление.
    easing_strength: float = 1.5


# Три готовых профиля. Добавляй свои по тому же шаблону.
PROFILES = {
    # Быстрый, уверенный пользователь: прямые короткие дуги, мало дрожания,
    # редкий перелёт, крупный пиксельный шаг (= высокая скорость).
    "fast_confident": MovementProfile(
        name="fast_confident",
        steps_per_100px=8,
        min_steps=8,
        max_steps=45,
        curve_amplitude=0.07,
        curve_amplitude_jitter=0.04,
        jitter_px=0.6,
        overshoot_probability=0.12,
        overshoot_px=(2.0, 8.0),
        step_delay=(0.004, 0.010),
        pre_click_pause=(0.05, 0.11),
        easing_strength=1.3,
    ),
    # Усреднённый пользователь.
    "average": MovementProfile(
        name="average",
        steps_per_100px=14,
        min_steps=12,
        max_steps=70,
        curve_amplitude=0.13,
        curve_amplitude_jitter=0.07,
        jitter_px=1.4,
        overshoot_probability=0.28,
        overshoot_px=(4.0, 16.0),
        step_delay=(0.008, 0.020),
        pre_click_pause=(0.07, 0.15),
        easing_strength=1.8,
    ),
    # Медленный, сомневающийся: широкие "гуляющие" дуги, сильное дрожание,
    # частый перелёт с коррекцией, мелкий шаг (= низкая скорость), длинные паузы.
    "slow_hesitant": MovementProfile(
        name="slow_hesitant",
        steps_per_100px=24,
        min_steps=20,
        max_steps=120,
        curve_amplitude=0.22,
        curve_amplitude_jitter=0.12,
        jitter_px=2.6,
        overshoot_probability=0.45,
        overshoot_px=(8.0, 28.0),
        step_delay=(0.015, 0.040),
        pre_click_pause=(0.11, 0.15),
        easing_strength=2.4,
    ),
}


# --------------------------------------------------------------------------- #
#  Основной класс
# --------------------------------------------------------------------------- #
class HumanCursor:
    """Человекоподобное управление курсором поверх Playwright Page."""

    def __init__(
        self,
        page: Page,
        profile: MovementProfile = PROFILES["average"],
        start: Point = (0.0, 0.0),
        rng: Optional[random.Random] = None,
    ) -> None:
        self.page = page
        self.profile = profile
        self._x, self._y = start
        # Передай seed для воспроизводимых фикстур: random.Random(42)
        self._rng = rng or random.Random()

    # -- Публичное состояние ------------------------------------------------ #
    @property
    def position(self) -> Point:
        return (self._x, self._y)

    def set_profile(self, profile: MovementProfile) -> None:
        self.profile = profile

    # -- Математика кривой -------------------------------------------------- #
    @staticmethod
    def _cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
        mt = 1.0 - t
        a = mt * mt * mt
        b = 3 * mt * mt * t
        c = 3 * mt * t * t
        d = t * t * t
        x = a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0]
        y = a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]
        return (x, y)

    def _ease(self, t: float) -> float:
        """Ремаппинг t для неравномерной скорости (разгон/замедление у концов)."""
        k = self.profile.easing_strength
        if k == 1.0:
            return t
        num = t ** k
        return num / (num + (1.0 - t) ** k)

    def _control_points(self, p0: Point, p3: Point) -> Tuple[Point, Point]:
        """Контрольные точки с перпендикулярным отклонением.

        Знак и величина отклонения рандомизируются независимо для каждой из
        двух точек -> дуги не повторяются от вызова к вызову.
        """
        dx, dy = p3[0] - p0[0], p3[1] - p0[1]
        dist = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / dist, dx / dist  # единичная нормаль к отрезку

        p = self.profile
        amp = dist * (
            p.curve_amplitude
            + self._rng.uniform(-p.curve_amplitude_jitter, p.curve_amplitude_jitter)
        )

        off1 = amp * self._rng.uniform(0.5, 1.0) * self._rng.choice((-1, 1))
        off2 = amp * self._rng.uniform(0.5, 1.0) * self._rng.choice((-1, 1))

        p1 = (p0[0] + dx / 3 + nx * off1, p0[1] + dy / 3 + ny * off1)
        p2 = (p0[0] + 2 * dx / 3 + nx * off2, p0[1] + 2 * dy / 3 + ny * off2)
        return p1, p2

    def _sample_count(self, distance: float) -> int:
        raw = int(distance / 100.0 * self.profile.steps_per_100px)
        return max(self.profile.min_steps, min(self.profile.max_steps, raw))

    # -- Низкоуровневое скольжение ------------------------------------------ #
    def _glide(self, target: Point) -> None:
        """Проводит курсор из текущей точки в target по одной кривой Безье."""
        p0 = (self._x, self._y)
        p3 = target
        dist = math.hypot(p3[0] - p0[0], p3[1] - p0[1])

        # Совсем короткий путь — двигаемся напрямую, без дуги.
        if dist < 4.0:
            self.page.mouse.move(p3[0], p3[1])
            self._x, self._y = p3
            return

        p1, p2 = self._control_points(p0, p3)
        n = self._sample_count(dist)
        jitter = self.profile.jitter_px

        for i in range(1, n + 1):
            t = i / n
            te = self._ease(t)
            x, y = self._cubic_bezier(p0, p1, p2, p3, te)

            if i < n and jitter > 0:
                # Дрожание гаснет у концов (sin-окно), чтобы приземлиться точно.
                fade = math.sin(math.pi * t)
                x += self._rng.gauss(0.0, jitter) * fade
                y += self._rng.gauss(0.0, jitter) * fade

            self.page.mouse.move(x, y)
            time.sleep(self._rng.uniform(*self.profile.step_delay))

        # Финальная точка — ровно в цель (без дрожания).
        self.page.mouse.move(p3[0], p3[1])
        self._x, self._y = p3

    # -- Движение к координате --------------------------------------------- #
    def move_to(self, x: float, y: float, overshoot: Optional[bool] = None) -> None:
        """Подводит курсор к (x, y). При перелёте делает коррекцию."""
        do_overshoot = (
            overshoot
            if overshoot is not None
            else self._rng.random() < self.profile.overshoot_probability
        )

        if do_overshoot:
            mag = self._rng.uniform(*self.profile.overshoot_px)
            ang = self._rng.uniform(0, 2 * math.pi)
            miss = (x + math.cos(ang) * mag, y + math.sin(ang) * mag)
            self._glide(miss)                       # промах за цель
            time.sleep(self._rng.uniform(0.02, 0.06))
            self._glide((x, y))                     # короткая коррекция в цель
        else:
            self._glide((x, y))

    # -- Движение к локатору ------------------------------------------------ #
    def _target_in_locator(self, locator: Locator) -> Point:
        locator.scroll_into_view_if_needed()
        box = locator.bounding_box()
        if box is None:
            raise RuntimeError("Не удалось получить bounding_box локатора (элемент не виден)")

        # Целимся не ровно в центр, а в случайную точку центральной зоны —
        # так клики по одному элементу не ложатся в один пиксель.
        cx = box["x"] + box["width"] * self._rng.uniform(0.35, 0.65)
        cy = box["y"] + box["height"] * self._rng.uniform(0.35, 0.65)
        return (cx, cy)

    def move_to_locator(self, locator: Locator) -> Point:
        target = self._target_in_locator(locator)
        self.move_to(*target)
        return target

    # hover == подвести курсор без клика
    hover = move_to_locator

    # -- Безопасный клик ---------------------------------------------------- #
    def safe_click(self, locator: Locator, button: str = "left") -> None:
        """Наведение -> случайная пауза (по профилю) -> клик в текущей точке.

        Клик выполняется через mouse.down/up в точке, куда реально приехал
        курсор, поэтому вся траектория наведения остаётся в данных аналитики.
        """
        self.move_to_locator(locator)
        time.sleep(self._rng.uniform(*self.profile.pre_click_pause))
        self.page.mouse.down(button=button)
        time.sleep(self._rng.uniform(0.03, 0.09))  # естественная длительность нажатия
        self.page.mouse.up(button=button)


# --------------------------------------------------------------------------- #
#  Подключение к существующему инстансу по CDP
# --------------------------------------------------------------------------- #
@contextmanager
def connect_cdp(endpoint_url: str) -> Iterator[Browser]:
    """Подключается к уже запущенному браузеру (например, в Aerokube Moon).

    endpoint_url — ws:// или http:// CDP-эндпоинт существующего инстанса.
    Браузер не поднимается заново: берётся уже авторизованная сессия.
    """
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(endpoint_url)
        try:
            yield browser
        finally:
            # Рвём только CDP-соединение, сам удалённый инстанс не закрываем.
            browser.close()


def first_existing_page(browser: Browser) -> Page:
    """Достаёт первую страницу из уже существующего контекста."""
    if not browser.contexts:
        raise RuntimeError("У подключённого браузера нет контекстов")
    ctx = browser.contexts[0]
    if not ctx.pages:
        raise RuntimeError("В контексте нет открытых страниц")
    return ctx.pages[0]


# --------------------------------------------------------------------------- #
#  Пример использования
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    CDP_ENDPOINT = "ws://localhost:4444/devtools/..."  # замените на свой эндпоинт Moon

    with connect_cdp(CDP_ENDPOINT) as browser:
        page = first_existing_page(browser)

        # Фикстура №1: быстрый уверенный пользователь, с фиксированным seed.
        cursor = HumanCursor(
            page,
            profile=PROFILES["fast_confident"],
            start=(10, 10),
            rng=random.Random(42),
        )
        cursor.safe_click(page.get_by_role("button", name="Купить"))

        # Фикстура №2: переключаемся на "сомневающегося" для того же сценария.
        cursor.set_profile(PROFILES["slow_hesitant"])
        cursor.hover(page.get_by_text("Подробнее"))
        cursor.safe_click(page.get_by_role("link", name="Корзина"))

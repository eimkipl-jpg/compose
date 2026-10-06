"""
EngagementSimulator — генератор синтетического dwell time для стресс-теста RecSys.

Модель (смесь из двух компонент):
    1. Bounce (вероятность p_bounce): мгновенный скролл, Uniform(0.5, 1.5) сек.
       Контент полностью нерелевантен.
    2. Engaged (вероятность 1 - p_bounce): латентное «желание смотреть»
       T ~ LogNormal(mu_eff, sigma), у которого тяжёлый правый хвост.
         - T <  duration  -> PARTIAL:  dwell = T (ушёл, не досмотрев)
         - T >= duration  -> COMPLETE: dwell = duration (досмотрел до конца)
             и с вероятностью rewatch_prob видео пошло на повтор ->
             REWATCH: dwell = duration * U(rewatch_multiplier_range)

Динамические поправки (v2):
    p_bounce и mu больше не константы, а функции от контекста показа.

    Bounce считается в логит-пространстве, поэтому поправки складываются
    аддитивно, а вероятность гарантированно остаётся в (0, 1):

        logit(p_bounce) = logit(bounce_prob)                 <- база профиля
                        + length_shift(video_duration_sec)   <- длина ролика
                        + fatigue_bounce_logit * F(i)        <- усталость

        mu_eff = mu - fatigue_mu_drop * F(i)

    length_shift(d) = -A * tanh( ln(d / d_ref) / (2 * s) )
        Это центрированная логистика по log-длительности. Она убывает,
        ограничена в [-A, +A] и равна нулю при d = d_ref, то есть на
        «референсной» длине p_bounce в точности равен базе профиля.

    F(i) = (σ((i - onset)/width) - σ(-onset/width)) / (1 - σ(-onset/width))
        Нормированная логистика по индексу ролика в сессии (0-based).
        F(0) = 0 ровно, плавный рост в районе onset, насыщение к 1.

Параметры лог-нормального распределения:
    mode   = exp(mu - sigma^2)   <- «горб», зона быстрых скипов
    median = exp(mu)
    mean   = exp(mu + sigma^2 / 2)
Чтобы задать когорту в понятных секундах, удобнее использовать
EngagementProfile.from_mode_median(...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Outcome(IntEnum):
    BOUNCE = 0    # мгновенный отказ
    PARTIAL = 1   # ушёл до конца
    COMPLETE = 2  # досмотрел
    REWATCH = 3   # досмотрел и пошёл на повтор


def _sigmoid(x: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# --------------------------------------------------------------------------- #
#  Конфиги динамических эффектов
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FeedFatigue:
    """Усталость от ленты: по мере роста индекса ролика в сессии
    растёт bounce и падает желание смотреть (mu)."""

    onset: float = 18.0               # индекс ролика, где рост идёт быстрее всего
    width: float = 4.0                # ширина перехода (в роликах); меньше -> резче
    bounce_logit_shift: float = 1.5   # прирост logit(p_bounce) при полной усталости
    mu_drop: float = 0.5              # падение mu при полной усталости (медиана × e^-0.5 ≈ ×0.61)

    def __post_init__(self) -> None:
        if self.onset < 0:
            raise ValueError("onset must be >= 0")
        if self.width <= 0:
            raise ValueError("width must be > 0")
        if self.bounce_logit_shift < 0 or self.mu_drop < 0:
            raise ValueError("fatigue effects must be >= 0 (fatigue only worsens engagement)")

    def level(self, session_indices: NDArray[np.float64]) -> NDArray[np.float64]:
        """F(i) ∈ [0, 1): нормирована так, что F(0) == 0."""
        s0 = _sigmoid(-self.onset / self.width)
        s = _sigmoid((session_indices - self.onset) / self.width)
        return (s - s0) / (1.0 - s0)


@dataclass(frozen=True)
class LengthBounce:
    """Зависимость мгновенного отказа от длины ролика: короткие скипают
    чаще, длинные реже. На ref_duration_sec поправка равна нулю."""

    ref_duration_sec: float = 30.0   # длина, на которой p_bounce == bounce_prob профиля
    amplitude: float = 1.2           # максимальный сдвиг logit(p_bounce) в обе стороны
    scale: float = 1.0               # «мягкость» по ln(длительности)

    def __post_init__(self) -> None:
        if self.ref_duration_sec <= 0 or self.scale <= 0:
            raise ValueError("ref_duration_sec and scale must be > 0")
        if self.amplitude < 0:
            raise ValueError("amplitude must be >= 0")

    def logit_shift(self, durations: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.log(durations / self.ref_duration_sec) / self.scale
        return -self.amplitude * np.tanh(x / 2.0)


# --------------------------------------------------------------------------- #
#  Профиль когорты
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EngagementProfile:
    """Поведенческий профиль когорты пользователей.

    fatigue / length_bounce = None выключает соответствующий эффект.
    """

    name: str
    mu: float                     # параметр LogNormal (лог-шкала, секунды)
    sigma: float                  # «толщина хвоста»; больше -> тяжелее хвост
    bounce_prob: float = 0.15     # база: p_bounce на ref-длине в начале сессии
    bounce_range: tuple[float, float] = (0.5, 1.5)
    rewatch_prob: float = 0.08    # P(повтор | досмотрел)
    rewatch_multiplier_range: tuple[float, float] = (1.1, 1.5)
    min_dwell_sec: float = 0.2    # минимальный фиксируемый трекером тайминг
    fatigue: FeedFatigue | None = field(default_factory=FeedFatigue)
    length_bounce: LengthBounce | None = field(default_factory=LengthBounce)

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        # bounce_prob строго внутри (0, 1): иначе logit не определён
        if not 0.0 < self.bounce_prob < 1.0:
            raise ValueError(f"bounce_prob must be in (0, 1), got {self.bounce_prob}")
        if not 0.0 <= self.rewatch_prob <= 1.0:
            raise ValueError(f"rewatch_prob must be in [0, 1], got {self.rewatch_prob}")
        lo, hi = self.bounce_range
        if not 0 < lo <= hi:
            raise ValueError("bounce_range must satisfy 0 < lo <= hi")
        lo, hi = self.rewatch_multiplier_range
        if not 1.0 < lo <= hi:
            raise ValueError("rewatch_multiplier_range must satisfy 1 < lo <= hi")
        if self.min_dwell_sec < 0:
            raise ValueError("min_dwell_sec must be >= 0")

    @classmethod
    def from_mode_median(
        cls, name: str, mode_sec: float, median_sec: float, **kwargs
    ) -> "EngagementProfile":
        """Задать когорту через интуитивные величины: где «горб» и где медиана.

        mu = ln(median), sigma = sqrt(ln(median) - ln(mode)).
        Чем дальше медиана от моды, тем тяжелее хвост.
        """
        if not 0 < mode_sec < median_sec:
            raise ValueError("require 0 < mode_sec < median_sec")
        mu = math.log(median_sec)
        sigma = math.sqrt(mu - math.log(mode_sec))
        return cls(name=name, mu=mu, sigma=sigma, **kwargs)

    # Справочные характеристики латентного распределения в начале сессии
    @property
    def mode(self) -> float:
        return math.exp(self.mu - self.sigma**2)

    @property
    def median(self) -> float:
        return math.exp(self.mu)

    @property
    def mean(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2)


# Готовые когорты: горб у обеих лежит в зоне 1–4 сек, хвосты разные.
# Нетерпеливые устают раньше и сильнее реагируют на длину ролика.
IMPATIENT = EngagementProfile.from_mode_median(
    "impatient", mode_sec=1.5, median_sec=3.5,
    bounce_prob=0.25, rewatch_prob=0.03,
    fatigue=FeedFatigue(onset=12, width=3, bounce_logit_shift=1.8, mu_drop=0.6),
    length_bounce=LengthBounce(ref_duration_sec=30, amplitude=1.5),
)
LOYAL = EngagementProfile.from_mode_median(
    "loyal", mode_sec=3.5, median_sec=12.0,
    bounce_prob=0.08, rewatch_prob=0.10,
    fatigue=FeedFatigue(onset=25, width=5, bounce_logit_shift=1.2, mu_drop=0.4),
    length_bounce=LengthBounce(ref_duration_sec=30, amplitude=0.8),
)


# --------------------------------------------------------------------------- #
#  Симулятор
# --------------------------------------------------------------------------- #

class EngagementSimulator:
    """Сэмплер dwell time для заданного профиля.

    Вся генерация векторизована: sample_batch на миллион показов
    занимает доли секунды. Для воспроизводимости передайте seed
    или общий np.random.Generator.
    """

    def __init__(
        self,
        profile: EngagementProfile,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        self.profile = profile
        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self._base_bounce_logit = _logit(profile.bounce_prob)

    # ---------------------------- публичный API ---------------------------- #

    def sample(self, video_duration_sec: float, session_index: int | None = None) -> float:
        """Один показ карточки -> dwell time в секундах."""
        idx = None if session_index is None else [session_index]
        return float(self.sample_batch([video_duration_sec], session_indices=idx)[0])

    def sample_session(
        self, video_durations_sec: ArrayLike, return_outcomes: bool = False
    ):
        """Одна сессия: ролики идут подряд, индексы 0..n-1 проставляются сами."""
        d = np.asarray(video_durations_sec, dtype=np.float64).ravel()
        return self.sample_batch(d, session_indices=np.arange(d.size),
                                 return_outcomes=return_outcomes)

    def effective_params(
        self,
        video_durations_sec: ArrayLike,
        session_indices: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(p_bounce, mu_eff) для каждого показа — без сэмплирования.

        Удобно для юнит-тестов и для графиков «как профиль деградирует».
        """
        d, idx, _ = self._prepare_inputs(video_durations_sec, session_indices)
        return self._compute_params(d, idx)

    def sample_batch(
        self,
        video_durations_sec: ArrayLike,
        session_indices: ArrayLike | None = None,
        return_outcomes: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.int8]]:
        """Векторная генерация: массив длительностей -> массив dwell time.

        session_indices — порядковые номера роликов в сессии (0-based),
            той же формы, что и durations (или скаляр, он будет растянут).
            None = все показы в начале сессии, без усталости.
        return_outcomes=True дополнительно возвращает метки Outcome.
        """
        d, idx, shape = self._prepare_inputs(video_durations_sec, session_indices)
        p_bounce, mu_eff = self._compute_params(d, idx)

        prof, rng, n = self.profile, self._rng, d.size
        dwell = np.empty(n, dtype=np.float64)
        outcomes = np.empty(n, dtype=np.int8)

        # 1. Bounce-компонента: у каждого показа своя вероятность
        is_bounce = rng.random(n) < p_bounce
        n_bounce = int(is_bounce.sum())
        dwell[is_bounce] = np.minimum(
            rng.uniform(*prof.bounce_range, size=n_bounce), d[is_bounce]
        )
        outcomes[is_bounce] = Outcome.BOUNCE

        # 2. Engaged-компонента: LogNormal(mu_eff_i, sigma) = exp(mu_eff_i + sigma * Z)
        engaged = ~is_bounce
        n_eng = n - n_bounce
        dur = d[engaged]
        latent = np.exp(mu_eff[engaged] + prof.sigma * rng.standard_normal(n_eng))

        reached_end = latent >= dur
        rewatch = reached_end & (rng.random(n_eng) < prof.rewatch_prob)
        multiplier = rng.uniform(*prof.rewatch_multiplier_range, size=n_eng)

        watched = np.minimum(np.maximum(latent, prof.min_dwell_sec), dur)
        dwell[engaged] = np.where(rewatch, dur * multiplier, watched)
        outcomes[engaged] = np.select(
            [rewatch, reached_end],
            [Outcome.REWATCH, Outcome.COMPLETE],
            default=Outcome.PARTIAL,
        )

        dwell = dwell.reshape(shape)
        if return_outcomes:
            return dwell, outcomes.reshape(shape)
        return dwell

    # ------------------------------ внутреннее ----------------------------- #

    @staticmethod
    def _prepare_inputs(
        video_durations_sec: ArrayLike, session_indices: ArrayLike | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, tuple[int, ...]]:
        durations = np.asarray(video_durations_sec, dtype=np.float64)
        shape = durations.shape
        d = durations.ravel()
        if d.size == 0:
            raise ValueError("video_durations_sec is empty")
        if not np.all(np.isfinite(d)) or np.any(d <= 0):
            raise ValueError("video durations must be finite and > 0")

        if session_indices is None:
            return d, None, shape
        idx_arr = np.asarray(session_indices, dtype=np.float64)
        try:
            idx = np.broadcast_to(idx_arr, shape).ravel()
        except ValueError as exc:
            raise ValueError(
                f"session_indices shape {idx_arr.shape} does not match durations {shape}"
            ) from exc
        if not np.all(np.isfinite(idx)) or np.any(idx < 0):
            raise ValueError("session_indices must be finite and >= 0")
        return d, idx, shape

    def _compute_params(
        self, d: NDArray[np.float64], idx: NDArray[np.float64] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        prof = self.profile
        bounce_logit = np.full(d.size, self._base_bounce_logit)
        mu_eff = np.full(d.size, prof.mu)

        if prof.length_bounce is not None:
            bounce_logit += prof.length_bounce.logit_shift(d)

        if prof.fatigue is not None and idx is not None:
            f = prof.fatigue.level(idx)
            bounce_logit += prof.fatigue.bounce_logit_shift * f
            mu_eff -= prof.fatigue.mu_drop * f

        return _sigmoid(bounce_logit), mu_eff


if __name__ == "__main__":
    n = 200_000

    for profile in (IMPATIENT, LOYAL):
        sim = EngagementSimulator(profile, seed=42)
        print(f"\n[{profile.name}] mu={profile.mu:.3f} sigma={profile.sigma:.3f} "
              f"base bounce={profile.bounce_prob:.0%}")

        # Length-dependent bounce (начало сессии)
        durs = np.array([5, 10, 15, 30, 60, 120, 300], dtype=float)
        p, _ = sim.effective_params(durs)
        print("  p_bounce по длине:   " +
              "  ".join(f"{int(x)}s={v:.1%}" for x, v in zip(durs, p)))

        # Feed fatigue на ролике 30 с
        idxs = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60], dtype=float)
        p, mu = sim.effective_params(np.full(idxs.size, 30.0), idxs)
        print("  p_bounce по индексу: " +
              "  ".join(f"#{int(i)}={v:.1%}" for i, v in zip(idxs, p)))
        print("  медиана T по индексу:" +
              "  ".join(f"#{int(i)}={np.exp(m):.1f}s" for i, m in zip(idxs, mu)))

        # Эмпирическая проверка: сэмпл совпадает с теорией
        for i in (0, 30):
            dwell, out = sim.sample_batch(np.full(n, 30.0), session_indices=i,
                                          return_outcomes=True)
            exp_p, _ = sim.effective_params([30.0], [i])
            print(f"  сэмпл idx={i:>2}: bounce={np.mean(out == Outcome.BOUNCE):.1%} "
                  f"(теория {exp_p[0]:.1%})  p50={np.median(dwell):.2f}s "
                  f"p90={np.percentile(dwell, 90):.2f}s")

    # Смешанная выборка: реальные сессии разной длины
    rng = np.random.default_rng(0)
    sim = EngagementSimulator(LOYAL, seed=1)
    lens = rng.integers(1, 60, size=5_000)
    durs = rng.lognormal(np.log(25), 0.8, size=lens.sum()).clip(3, 600)
    idx = np.concatenate([np.arange(k) for k in lens])
    dwell = sim.sample_batch(durs, session_indices=idx)
    print(f"\n[mixed loyal] показов={dwell.size}, "
          f"уникальных значений={len(np.unique(dwell.round(2)))}")

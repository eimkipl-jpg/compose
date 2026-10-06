"""
detector_features.py
=====================

Детекторная сторона: извлечение дискриминирующих признаков из указательной
телеметрии (x, y, t). Все функции — чистые, работают над numpy-массивами,
никакой привязки к браузеру/Playwright/курсору.

Идея: разложить траекторию на производные и спектральные представления и
померить то, что синтетика воспроизводит плохо. Признаки сгруппированы по
трём семействам:

  A. Спектр/корреляция шума  — отличить коррелированный шум (fBm/Perlin)
     от белого гауссова и, что важнее, поймать артефакты самого Perlin.
  B. Физика ввода            — квантование координат и таймингов опроса,
     физиологический тремор. Это признаки, которые трудно подделать даже
     зная о них.
  C. Idle / пассивные фазы   — моменты, когда курсор "гуляет" без цели.

Зависимости: numpy, scipy.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


# --------------------------------------------------------------------------- #
#  Базовые кинематические производные
# --------------------------------------------------------------------------- #
def kinematics(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict:
    """Скорость, ускорение, рывок (jerk), щелчок (snap) по телеметрии.

    t — временные метки в секундах (неравномерные допустимы).
    Возвращает словарь массивов; синтетика обычно заметна в высших
    производных (jerk/snap), т.к. сглаженные кривые Безье дают неестественно
    "чистые" распределения рывка.
    """
    dt = np.diff(t)
    dt = np.where(dt <= 0, 1e-6, dt)

    dx, dy = np.diff(x), np.diff(y)
    speed = np.hypot(dx, dy) / dt

    accel = np.diff(speed) / dt[1:]
    jerk = np.diff(accel) / dt[2:]
    snap = np.diff(jerk) / dt[3:]

    return {"dt": dt, "speed": speed, "accel": accel, "jerk": jerk, "snap": snap}


def distribution_moments(arr: np.ndarray) -> dict:
    """Моменты распределения. У живого сигнала рывок тяжелохвостый и
    асимметричный; у сглаженной синтетики — подозрительно близок к норме.
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return {"std": np.nan, "skew": np.nan, "kurtosis": np.nan}
    m = arr.mean()
    s = arr.std()
    if s == 0:
        return {"std": 0.0, "skew": 0.0, "kurtosis": 0.0}
    z = (arr - m) / s
    return {
        "std": float(s),
        "skew": float(np.mean(z ** 3)),
        "kurtosis": float(np.mean(z ** 4) - 3.0),  # эксцесс (0 = норма)
    }


# --------------------------------------------------------------------------- #
#  A. Спектр и корреляция шума
# --------------------------------------------------------------------------- #
def psd_slope(sig: np.ndarray, fs: float) -> float:
    """Наклон β спектральной плотности мощности в log-log (PSD ~ 1/f^β).

      белый гауссов шум  -> β ≈ 0  (плоский спектр)
      fBm (Hurst H)      -> β ≈ 2H + 1
      Perlin/градиентный -> характерный спад + артефакты решётки

    Оценка по средней полосе частот (без DC и окрестности Найквиста).
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    n = sig.size
    if n < 16:
        return np.nan

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(sig)) ** 2

    mask = (freqs > 0) & (freqs < fs / 2.0)
    f, p = freqs[mask], power[mask]
    lo, hi = int(0.05 * f.size), int(0.9 * f.size)  # обрезаем края
    f, p = f[lo:hi], p[lo:hi]
    p = np.where(p <= 0, 1e-12, p)

    slope, _ = np.polyfit(np.log(f), np.log(p), 1)
    return float(-slope)  # знак: положительный β = "краснее" белого


def spectral_flatness(sig: np.ndarray) -> float:
    """Плоскостность спектра (мера Винера), 0..1.

      ~1.0 -> близко к белому шуму (гаусс)
      <<1  -> выраженная окраска (коррелированный шум)
    """
    sig = np.asarray(sig, dtype=float) - np.mean(sig)
    power = np.abs(np.fft.rfft(sig)) ** 2
    power = power[1:]  # без DC
    power = np.where(power <= 0, 1e-12, power)
    gmean = np.exp(np.mean(np.log(power)))
    amean = np.mean(power)
    return float(gmean / amean)


def autocorr(sig: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Нормированная автокорреляция. Белый шум ~0 на лагах >0;
    коррелированный шум спадает медленно.
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    full = np.correlate(sig, sig, mode="full")
    mid = full.size // 2
    ac = full[mid: mid + max_lag + 1]
    return ac / (ac[0] + 1e-12)


def hurst_rs(sig: np.ndarray) -> float:
    """Оценка показателя Хёрста методом R/S.

      H ≈ 0.5 -> независимые приращения (близко к гауссову шуму)
      H > 0.5 -> долгая память (fBm, "органический" дрейф)
    """
    sig = np.asarray(sig, dtype=float)
    n = sig.size
    if n < 64:
        return np.nan
    scales = np.unique(np.floor(np.logspace(2, np.log10(n // 2), 12)).astype(int))
    rs = []
    for s in scales:
        if s < 8:
            continue
        chunks = n // s
        vals = []
        for i in range(chunks):
            seg = sig[i * s:(i + 1) * s]
            z = seg - seg.mean()
            cum = np.cumsum(z)
            r = cum.max() - cum.min()
            sd = seg.std()
            if sd > 0:
                vals.append(r / sd)
        if vals:
            rs.append((s, np.mean(vals)))
    if len(rs) < 3:
        return np.nan
    s_arr = np.log([p[0] for p in rs])
    rs_arr = np.log([p[1] for p in rs])
    h, _ = np.polyfit(s_arr, rs_arr, 1)
    return float(h)


def lattice_periodicity(sig: np.ndarray, max_lag: int = 80) -> float:
    """Детектор артефактов решётки (ахиллесова пята Perlin/градиентного шума).

    Градиентный шум интерполирует между узлами целочисленной решётки и
    строго зануляется в узлах -> в автокорреляции появляются регулярные
    пики с фиксированным шагом. Чем регулярнее пики, тем выше скор.

      высокий скор -> вероятен решётчатый (Perlin-подобный) генератор
      низкий скор  -> гаусс или настоящий тремор
    """
    ac = autocorr(sig, max_lag)[1:]
    peaks, _ = sp_signal.find_peaks(ac)
    if peaks.size < 3:
        return 0.0
    gaps = np.diff(peaks)
    regularity = 1.0 - (gaps.std() / (gaps.mean() + 1e-9))
    return float(max(0.0, regularity))


# --------------------------------------------------------------------------- #
#  B. Физика ввода — самые устойчивые признаки
# --------------------------------------------------------------------------- #
def coordinate_quantization(x: np.ndarray, y: np.ndarray) -> float:
    """Насколько координаты лежат на целочисленной решётке устройства.

    Реальные события мыши приходят в целых device-пикселях. Сгенерированные
    float-координаты (из Безье) после округления дают иную структуру остатка.

      ~0  -> значения почти целые (похоже на настоящее железо)
      >0  -> дробные остатки (след синтетики, если данные сырые, не сглаженные)
    """
    rx = np.abs(x - np.round(x))
    ry = np.abs(y - np.round(y))
    return float(np.mean(np.concatenate([rx, ry])))


def polling_rate_signature(t: np.ndarray) -> dict:
    """Структура межсобытийных интервалов (опрос устройства).

    Настоящая мышь опрашивается на 125/500/1000 Гц -> dt группируется у
    8/2/1 мс. Синтетические паузы (uniform/gauss sleep) такой решётки не дают.
    Возвращает ближайшую стандартную частоту и долю интервалов рядом с ней.
    """
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return {"best_hz": np.nan, "lock_fraction": 0.0, "dt_cv": np.nan}

    candidates = {125: 1 / 125, 250: 1 / 250, 500: 1 / 500, 1000: 1 / 1000}
    best_hz, best_frac = np.nan, 0.0
    for hz, period in candidates.items():
        near = np.mean(np.abs(dt - period) < 0.15 * period)
        if near > best_frac:
            best_hz, best_frac = hz, near

    return {
        "best_hz": best_hz,
        "lock_fraction": float(best_frac),      # высокая -> настоящее железо
        "dt_cv": float(dt.std() / (dt.mean() + 1e-9)),
    }


def tremor_band_energy(speed: np.ndarray, fs: float,
                       band=(8.0, 12.0)) -> float:
    """Доля энергии в полосе физиологического тремора кисти (~8–12 Гц).

    У живой руки в этой полосе всегда есть остаточная энергия даже при
    "удержании". Чистая математическая кривая её не содержит.

      >0 заметно -> присутствует тремор (человек)
      ~0         -> подозрительно гладко (синтетика)
    """
    speed = np.asarray(speed, dtype=float)
    if speed.size < 32 or fs <= 2 * band[1]:
        return np.nan
    f, pxx = sp_signal.welch(speed - speed.mean(), fs=fs,
                             nperseg=min(256, speed.size))
    total = np.trapz(pxx, f) + 1e-12
    bmask = (f >= band[0]) & (f <= band[1])
    return float(np.trapz(pxx[bmask], f[bmask]) / total)


# --------------------------------------------------------------------------- #
#  C. Idle / пассивные фазы
# --------------------------------------------------------------------------- #
def idle_features(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                  move_eps: float = 1.5) -> dict:
    """Признаки пассивного поведения ("чтение", бесцельные микросдвиги).

    Во время потребления контента человек двигает мышь мало, но не нулём:
    микродрейф + редкие дискретные сдвиги. Синтетика обычно либо стоит
    идеально (dwell слишком "мёртвый"), либо дрейфует слишком гладко.
    """
    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    moving = step > move_eps

    # Длины серий покоя между движениями.
    dwell_runs, run = [], 0
    for m in moving:
        if m:
            if run:
                dwell_runs.append(run)
            run = 0
        else:
            run += 1
    if run:
        dwell_runs.append(run)

    micro = step[(step > 0) & (step <= move_eps)]  # амплитуды микросдвигов

    return {
        "stationary_fraction": float(np.mean(~moving)),
        "dwell_run_cv": float(np.std(dwell_runs) / (np.mean(dwell_runs) + 1e-9))
        if dwell_runs else np.nan,
        "micro_amp_mean": float(micro.mean()) if micro.size else 0.0,
        "micro_amp_skew": distribution_moments(micro)["skew"]
        if micro.size >= 4 else np.nan,
    }


# --------------------------------------------------------------------------- #
#  Сводка
# --------------------------------------------------------------------------- #
def summarize(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict:
    """Единый вектор признаков по одной сессии телеметрии."""
    x, y, t = map(lambda a: np.asarray(a, dtype=float), (x, y, t))
    fs = 1.0 / (np.median(np.diff(t)) + 1e-9)
    k = kinematics(x, y, t)

    feats = {"fs_est": fs}

    # A
    feats["psd_slope_speed"] = psd_slope(k["speed"], fs)
    feats["spectral_flatness_speed"] = spectral_flatness(k["speed"])
    feats["hurst_speed"] = hurst_rs(k["speed"])
    feats["lattice_speed"] = lattice_periodicity(k["speed"])
    feats.update({f"jerk_{m}": v for m, v in distribution_moments(k["jerk"]).items()})
    feats.update({f"snap_{m}": v for m, v in distribution_moments(k["snap"]).items()})

    # B
    feats["coord_quantization"] = coordinate_quantization(x, y)
    feats.update({f"poll_{m}": v for m, v in polling_rate_signature(t).items()})
    feats["tremor_energy"] = tremor_band_energy(k["speed"], fs)

    # C
    feats.update({f"idle_{m}": v for m, v in idle_features(x, y, t).items()})

    return feats


if __name__ == "__main__":
    # Демо на случайных данных — замените на реальные сессии.
    rng = np.random.default_rng(0)
    n = 2000
    t = np.cumsum(rng.uniform(0.004, 0.02, n))
    x = np.cumsum(rng.normal(0, 2, n))
    y = np.cumsum(rng.normal(0, 2, n))
    from pprint import pprint
    pprint(summarize(x, y, t))

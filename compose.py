"""Универсальная сборка ролика по рецепту (из мастера «Создать»).

meta["recipe"] = {
  "video":  {"kind": "bg|clips|own", "theme": str|list|None, "balance": bool, "clips": int, "seconds": int, "file": str|None,
             "start": float|None, "end": float|None},
  "speech": {"kind": "none|audio|text|translate", "target": "en", "voice": str|None},
  "style":  "none|random|dynamic|calm|retro|mono",
  "look":   {layout, speed, fx, sub, subpos, hook, cta, logo}   - см. common/looks.py
  "sfx":    {cuts, start, tag, vol}                                - звуковые эффекты
}
Музыка/громкость/субтитры лежат в meta (music, music_vol, subs, lang) - это настройки профиля канала.
"""
import logging
import random
import re
import shutil
import subprocess
from pathlib import Path

from common import config, library
from common.looks import FRAME_HEX, FRAME_PX, HOOK_PRESETS, merged_look, merged_sfx
from common.media import audio_duration, run, video_duration
from core import dub, mix, render, subtitles, tts

log = logging.getLogger("compose")

TRANSITIONS = ["fade", "slideleft", "slideright", "wipeleft", "circleopen", "pixelize", "smoothleft", "radial", "dissolve"]

from core import montage

GRADES = montage.GRADES
FX_FILTERS = montage.FX_FILTERS
FRAME_COLORS = ["white", "black", "0x1b1b2f", "0xffd60a", "0xe63946"]
# качество итогового файла: CRF меньше - лучше картинка (18 почти без потерь), потолок битрейта защищает от раздувания
VIDEO_CRF, PART_CRF, MAXRATE = 18, 15, "12M"   # промежуточные куски с запасом: каждое перекодирование добавляет мягкости
# анонимность: ни одного тега ПО и автора, ни следа x264 в потоке (SEI с версией и настройками), пустые имена обработчиков
ANON_FLAGS = ["-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
              "-bsf:v", "filter_units=remove_types=6", "-metadata:s:v:0", "handler_name=", "-metadata:s:a:0", "handler_name=",
              "-metadata:s:v:0", "language=", "-metadata:s:a:0", "language=", "-metadata:s:v:0", "encoder=", "-metadata:s:a:0", "encoder="]


def scrub(path: Path) -> None:
    """Перепаковка без перекодирования: убирает теги кодировщика/muxer (Lavc, Lavf), даты, главы, метаданные потоков.
    Картинка и звук не меняются."""
    tmp = path.with_name(path.stem + "_clean.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-map", "0", "-c", "copy", "-map_metadata", "-1",
         "-map_metadata:s:v", "-1", "-map_metadata:s:a", "-1", "-map_chapters", "-1", "-fflags", "+bitexact",
         "-movflags", "+faststart", str(tmp)])
    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("не удалось очистить метаданные final.mp4")
    tmp.replace(path)


def _encode_variation() -> list[str]:
    """Небольшой разброс настроек кодирования между роликами: качество то же, но битовый поток каждый раз разный."""
    return ["-crf", str(random.choice([17, 18, 18, 19])), "-g", str(random.choice([60, 90, 120, 150, 240])),
            "-bf", str(random.choice([2, 3, 3, 4])), "-refs", str(random.choice([2, 3, 4]))]


LOUDNESS = "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.84:level=disabled,aresample=44100"  # норма YouTube, без клиппинга
SRC_RE = re.compile(r"_\d{4}-\d{4}$")   # готовая нарезка: <имя>_<начало>-<конец>.mp4
LOGO_XY = {"tr": "W-w-40:70", "tl": "40:70", "br": "W-w-40:H-h-300", "bl": "40:H-h-300"}

STYLES = montage.STYLES
STYLE_LABELS = montage.labels()   # снимок готовых стилей (для старых импортов); актуальный список, со своими: montage.labels()


def resolve_style(name: str) -> dict:
    """Конкретные параметры стиля (готового, своего или случайного) - см. core/montage.py."""
    return montage.resolve(name)


def _own_by_name(name: str, project: str | None = None) -> Path:
    if project:   # файлы проекта из банка идей ("канал/идея") ищем первыми
        base = (config.PROJECTS_DIR / project).resolve()
        cand = (base / name).resolve()
        if base.is_relative_to(config.PROJECTS_DIR.resolve()) and cand.parent == base and cand.is_file():
            return cand
    for f in mix.own_clips():
        if f.name == name:
            return f
    raise RuntimeError(f"в _own нет файла «{name}»")


def _silence(out: Path, length: float) -> None:
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", f"{length:.3f}", str(out)])


def _has_audio(clip: Path) -> bool:
    res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(clip)], capture_output=True, text=True)
    return bool(res.stdout.strip())


def _orig_audio(clip: Path, start: float, length: float, out: Path, speed: float = 1.0) -> bool:
    """Оригинальный звук окна клипа (length - длина на выходе; при speed окно в источнике длиннее).
    False, если у видео нет звуковой дорожки (тогда пишется тишина)."""
    if not _has_audio(clip):
        _silence(out, length)
        return False
    try:
        tempo = f"atempo={speed:.3f}," if abs(speed - 1.0) > 1e-3 else ""
        # громкость у клипов разная (тихий лай рядом с громким криком): выравниваем каждый до одного уровня
        level = "loudnorm=I=-18:TP=-2:LRA=11,aresample=44100,"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(clip),
             "-vn", "-ar", "44100", "-ac", "1", "-af", f"{tempo}{level}apad", "-t", f"{length:.3f}", str(out)])
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("не удалось извлечь звук из %s: %s", clip.name, str(e)[-200:])
        _silence(out, length)
        return False


# ---------- нарезка куска: компоновка кадра и скорость ----------

def _parse_frame(spec: str) -> tuple[int, str]:
    """spec вида «60:white» (толщина в пикселях : цвет или blur); без толщины берётся средняя."""
    px, _, col = spec.rpartition(":")
    return (int(px) if px.isdigit() else FRAME_PX["mid"]), (col or "white")


def frame_spec(look: dict) -> str:
    """Настройки рамки из look -> строка «толщина:цвет» для _cut_styled."""
    fr = look.get("frame") or {}
    px = FRAME_PX.get(fr.get("size", "mid"), FRAME_PX["mid"])
    col = fr.get("color", "random")
    if col == "random":
        col = random.choice(FRAME_COLORS)
    elif col == "blur":
        pass
    elif col in FRAME_HEX:
        col = FRAME_HEX[col]
    elif re.fullmatch(r"#?[0-9a-fA-F]{6}", col or ""):
        col = "0x" + col.lstrip("#")
    else:
        col = "white"
    return f"{px}:{col}"


def _layout_graph(layout: str, speed: float, color: str) -> str:
    """filter_complex для одного куска; результат - метка [v] размером 1080x1920, 30 fps."""
    sp = f"setpts=PTS/{speed:.3f}," if abs(speed - 1.0) > 1e-3 else ""
    tail = "setsar=1,fps=30"
    cover = "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920"
    if layout == "blur":
        return (f"[0:v]{sp}split[a][b];[a]{cover},boxblur=40:6,eq=brightness=-0.12[bg];"
                f"[b]scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{tail}[v]")
    if layout == "frame":
        px, col = _parse_frame(color)
        iw, ih = 1080 - 2 * px, 1920 - 2 * px
        inner = f"scale={iw}:{ih}:force_original_aspect_ratio=increase:flags=lanczos,crop={iw}:{ih}"
        if col == "blur":  # вместо цветной рамки - размытая копия самого клипа
            return (f"[0:v]{sp}split[a][b];[a]{cover},boxblur=40:6,eq=brightness=-0.12[bg];[b]{inner}[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{tail}[v]")
        return f"[0:v]{sp}{inner},pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color={col},{tail}[v]"
    if layout == "split":
        half = "scale=1080:960:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:960,setsar=1"
        return (f"[0:v]{sp}{half}[t];[1:v]{sp}{half}[b];[t][b]vstack=inputs=2,{tail}[v]")
    return f"[0:v]{sp}{cover},{tail}[v]"


def _cut_styled(clip: Path, start: float, length: float, out: Path, loop: bool = False, layout: str = "fill",
                speed: float = 1.0, clip2: Path | None = None, start2: float = 0.0, color: str = "60:white") -> None:
    """Кусок клипа без звука: 9:16, выбранная компоновка, скорость. length - длина на выходе."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           *(["-stream_loop", "-1"] if loop else []), "-ss", f"{start:.2f}", "-i", str(clip)]
    if layout == "split":
        second = clip2 or clip
        cmd += ["-stream_loop", "-1", "-ss", f"{start2:.2f}", "-i", str(second)]
    cmd += ["-t", f"{length:.2f}", "-an", "-filter_complex", _layout_graph(layout, speed, color), "-map", "[v]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(PART_CRF), "-pix_fmt", "yuv420p", str(out)]
    run(cmd)


def _pick_speed(look: dict) -> float:
    s = look.get("speed", "1")
    return random.choice([0.8, 1.0, 1.25, 1.5]) if s == "mix" else float(s)


def _second_pool(v: dict) -> list[Path]:
    """Кандидаты для нижней половины сплит-экрана."""
    try:
        return mix._clips(v.get("theme") or None) if v.get("kind") != "own" else mix._clips(None)
    except RuntimeError:
        return []


def _second_clip(clip: Path, pool: list[Path]) -> Path:
    return random.choice([c for c in pool if c != clip] or [clip])


# ---------- сбор видеоряда ----------

def _fixed_parts(work: Path, sources: list[tuple[Path, float]], seg, keep_audio: bool,
                 look: dict, pool2: list[Path] | None = None, color: str = "60:white", n0: int = 0):
    """seg - длина куска в секундах: одно число для всех или список по числу sources."""
    items = []
    layout = look.get("layout", "fill")
    for i, (clip, start) in enumerate(sources, n0):
        speed = _pick_speed(look)
        seg_i = seg[i - n0] if isinstance(seg, (list, tuple)) else seg
        need = seg_i * speed  # сколько секунд исходника потребуется
        dur = video_duration(clip)
        st = start if start is not None else (random.uniform(0, dur - need - 0.1) if dur > need + 0.1 else 0.0)
        clip2, st2 = None, 0.0
        if layout == "split":
            clip2 = _second_clip(clip, pool2 or [])
            d2 = video_duration(clip2)
            st2 = random.uniform(0, d2 - need - 0.1) if d2 > need + 0.1 else 0.0
        part = work / f"p{i:02}.mp4"
        _cut_styled(clip, st, seg_i, part, loop=dur - st <= need + 0.1, layout=layout, speed=speed,
                    clip2=clip2, start2=st2, color=color)
        item = {"part": part, "A": video_duration(part), "speed": speed}
        if keep_audio:
            wav = work / f"o{i:02}.wav"
            item["orig_real"] = _orig_audio(clip, st, item["A"], wav, speed)
            item["orig"] = wav
        items.append(item)
    return items


def _resolve_picks(v: dict) -> list[Path]:
    """Ролики, выбранные вручную (пути относительно ROOT), в порядке выбора. Пропавшие файлы отбрасываются."""
    out, missing = [], []
    for rel in v.get("picks") or []:
        f = config.ROOT / rel
        (out if f.is_file() else missing).append(f if f.is_file() else rel)
    if missing:
        log.warning("выбранные ролики не найдены: %s", ", ".join(map(str, missing[:5])))
    if v.get("picks") and not out:
        raise RuntimeError("выбранные ролики не найдены (возможно, удалены или не синхронизировались с Диском). Выберите заново.")
    return out


def video_height(path: Path) -> int:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip().split(",")[0])
    except (ValueError, IndexError, subprocess.SubprocessError):
        return 0


def theme_of_clip(c: Path) -> str:
    return c.parent.name.lower()


def _balance_cand(cand: list[Path], out: list) -> list[Path]:
    """Режим «поровну из тем»: из кандидатов оставляем клипы той темы, которой в ролике пока меньше всего."""
    if len({theme_of_clip(c) for c in cand}) < 2:
        return cand
    have: dict[str, int] = {}
    for item in out:
        t = theme_of_clip(item[0])
        have[t] = have.get(t, 0) + 1
    best = min(have.get(theme_of_clip(c), 0) for c in cand)
    return [c for c in cand if have.get(theme_of_clip(c), 0) == best]


def sample_balanced(pool: list[Path], n: int) -> list[Path]:
    """n клипов по кругу из разных тем (внутри темы случайно, без повторов, пока есть другие)."""
    def groups_of() -> dict[str, list[Path]]:
        g: dict[str, list[Path]] = {}
        for c in pool:
            g.setdefault(theme_of_clip(c), []).append(c)
        for lst in g.values():
            random.shuffle(lst)
        return g
    groups = groups_of()
    keys = list(groups)
    random.shuffle(keys)
    out: list[Path] = []
    while len(out) < n:
        took = False
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
                took = True
        if not took:
            groups = groups_of()
    return out


def filter_quality(pool: list[Path], min_h: int) -> list[Path]:
    """Оставляет клипы не ниже min_h пикселей по высоте. Если подходящих нет, возвращает всё (и пишет в лог)."""
    if not min_h:
        return pool
    good = [c for c in pool if video_height(c) >= min_h]
    if not good:
        log.warning("нет клипов от %sp, беру все", min_h)
        return pool
    return good


def plan_natural(pool: list[Path], total: float, exact: bool, balance: bool = False) -> list[tuple[Path, float]]:
    """Клипы целиком, каждый ровно своей длины. Набираем, пока не достигнем total; последний выбираем так, чтобы
    перебор был минимальным. exact=True (длина задана озвучкой): последний клип подрезается точно под total.
    Клипы не повторяются, соседние по возможности из разных исходников."""
    counts = library.use_counts(pool)
    order = list(pool)
    random.shuffle(order)
    order.sort(key=lambda c: counts.get(c, 0))
    durs = {c: min(video_duration(c) or 0.0, 60.0) for c in order}
    order = [c for c in order if durs[c] >= 0.5]
    if not order:
        raise RuntimeError("в выбранной теме нет подходящих клипов")
    out, acc, prev, used = [], 0.0, None, set()
    while acc < total - 0.3:
        avail = [c for c in order if c not in used]
        if not avail:
            used.clear()
            avail = [c for c in order if not out or c != out[-1][0]] or order
        cand = [c for c in avail if source_key(c) != prev] or avail
        if balance:
            cand = _balance_cand(cand, out)
        remain = total - acc
        if remain < 8:   # финиш: ищем клип, который закроет остаток с наименьшим перебором
            fit = [c for c in cand if durs[c] >= remain - 0.3]
            if fit:
                c = min(fit, key=lambda x: durs[x])
            else:
                c = cand[0]
        else:
            c = cand[0]
        length = durs[c]
        out.append((c, round(length, 2)))
        used.add(c)
        acc += length
        prev = source_key(c)
    if exact and out:   # подгон под озвучку: лишнее обрезаем, нехватку закрывает цикл последнего клипа
        over = acc - total
        c, length = out[-1]
        out[-1] = (c, round(max(length - over, 0.5), 2))
    return out


def natural_picks(picks: list[Path], total: float, exact: bool) -> list[float]:
    """Выбранные вручную клипы целиком. При exact подгоняем длину под озвучку (последний обрезаем или тянем)."""
    lens = [min(video_duration(c) or 1.0, 60.0) for c in picks]
    if exact:
        acc, out = 0.0, []
        for c, L in zip(picks, lens):
            take = min(L, max(total - acc, 0.0))
            if take < 0.4:
                break
            out.append(take)
            acc += take
        if out and acc < total - 0.1:
            out[-1] += total - acc
        lens = out
    return [round(x, 2) for x in lens]


def clip_range(v: dict) -> tuple[float, float] | None:
    """Диапазон длины клипа из рецепта («2-6») или None для старого режима «равные куски»."""
    s = str(v.get("len") or "eq")
    try:
        lo, hi = (float(x) for x in s.split("-"))
        return lo, max(hi, lo)
    except ValueError:
        return None


def source_key(clip: Path) -> str:
    """Исходное видео, из которого вырезан клип (для готовых нарезок), иначе имя файла."""
    return SRC_RE.sub("", clip.stem)


def _start_for(clip: Path) -> float | None:
    """Готовая нарезка - целая сцена, играем с её начала; длинное видео - случайное место."""
    return 0.0 if SRC_RE.search(clip.stem) else None


def plan_clips(pool: list[Path], total: float, lo: float, hi: float, balance: bool = False) -> list[tuple[Path, float]]:
    """Собирает ролик из готовых клипов: у каждого своя длина из [lo, hi], клипы не повторяются, пока есть другие,
    соседние клипы по возможности из разных исходников. Сумма длин = total. [(клип, длина)]."""
    counts = library.use_counts(pool)
    order = list(pool)
    random.shuffle(order)
    order.sort(key=lambda c: counts.get(c, 0))   # сначала те, что реже использовались
    durs = {c: (video_duration(c) or 0.0) for c in order}
    order = [c for c in order if durs[c] >= 0.8]
    if not order:
        raise RuntimeError("в выбранной теме нет клипов длиннее секунды")
    out, acc, prev, used = [], 0.0, None, set()
    while acc < total - 0.3:
        avail = [c for c in order if c not in used]
        if not avail:  # клипов меньше, чем нужно секунд: начинаем второй круг
            used.clear()
            avail = [c for c in order if not out or c != out[-1][0]] or order
        cand = [c for c in avail if source_key(c) != prev] or avail
        if balance:
            cand = _balance_cand(cand, out)
        c = cand[0]
        want = round(random.uniform(lo, hi), 1)
        length = min(durs[c], want)
        remain = total - acc
        if remain < length or remain - length < lo * 0.6:   # хвост короче минимума: отдаём его этому клипу
            length = min(durs[c], remain)
        if length < 0.5:
            break
        out.append((c, round(length, 2)))
        used.add(c)
        acc += length
        prev = source_key(c)
    if acc < total - 0.5:   # клипы кончились короче нужного: растянем последний (зацикливание в _cut_styled)
        c, length = out[-1]
        out[-1] = (c, round(length + total - acc, 2))
    return out


def fit_lengths(picks: list[Path], total: float, lo: float, hi: float) -> list[float]:
    """Длины для клипов, выбранных вручную: случайные из [lo, hi], затем подгон под total."""
    raw = [min(video_duration(c) or hi, random.uniform(lo, hi)) for c in picks]
    k = total / max(sum(raw), 0.1)
    return [round(max(x * k, 0.6), 2) for x in raw]


def _video_items(work: Path, v: dict, duration: float, keep_audio: bool, look: dict, color: str, exact: bool = False):
    kind = v["kind"]
    picks = _resolve_picks(v) if kind in ("clips", "own") and not v.get("file") else []
    rng = clip_range(v)
    full = v.get("len") == "full"
    if picks:  # ролики выбраны вручную: берём ровно их, в порядке выбора
        if full:
            seg = natural_picks(picks, duration, exact)
            picks = picks[:len(seg)]
        else:
            seg = fit_lengths(picks, duration, *rng) if rng else duration / len(picks)
        for c in picks:
            library.touch(c)
        return _fixed_parts(work, [(c, _start_for(c)) for c in picks], seg, keep_audio, look,
                            _second_pool(v) if look.get("layout") == "split" else None, color)
    pool2 = _second_pool(v) if look.get("layout") == "split" else None
    if kind == "bg":
        bg = render.pick_background([f"#{t}" for t in mix.norm_themes(v.get("theme"))])
        return _fixed_parts(work, [(bg, None)], duration, False, look, pool2, color)
    if kind == "clips" and (rng or full):   # компилятор готовых нарезок: без повторов, клипы своей или заданной длины
        pool = filter_quality(mix._clips(v.get("theme") or None), int(v.get("min_h") or 0))
        bal = bool(v.get("balance")) and len(mix.norm_themes(v.get("theme"))) > 1
        plan = plan_natural(pool, duration, exact, bal) if full else plan_clips(pool, duration, *rng, balance=bal)
        for c, _ in plan:
            library.touch(c)
        return _fixed_parts(work, [(c, _start_for(c)) for c, _ in plan], [L for _, L in plan], keep_audio, look, pool2, color)
    if kind == "clips":
        n = int(v.get("clips", 4))
        pool = mix._clips(v.get("theme") or None)
        if v.get("balance") and len(mix.norm_themes(v.get("theme"))) > 1:
            picked = sample_balanced(pool, n)
        else:
            picked = random.sample(pool, n) if len(pool) >= n else pool + random.choices(pool, k=n - len(pool))
            random.shuffle(picked)
        for c in picked:
            library.touch(c)
        return _fixed_parts(work, [(c, None) for c in picked], duration / n, keep_audio, look, pool2, color)
    if kind == "own":
        n = int(v.get("clips", 1))
        if v.get("file"):
            clip = _own_by_name(v["file"])
            library.touch(clip)
            start = v.get("start")
            if start is not None:
                length = (v["end"] - start) if v.get("end") else duration
            else:
                length = duration
            return _fixed_parts(work, [(clip, start)], max(length, 1.0), keep_audio, look, pool2, color)
        pool = mix.own_clips()
        if not pool:
            raise RuntimeError("в _own нет видео: положите свои ролики в ShortsFactory/_clips/_own/ на Диске")
        picked = random.sample(pool, n) if len(pool) >= n else pool + random.choices(pool, k=n - len(pool))
        for c in picked:
            library.touch(c)
        return _fixed_parts(work, [(c, None) for c in picked], duration / len(picked), keep_audio, look, pool2, color)
    raise RuntimeError(f"неизвестный видеоряд: {kind}")


def _translate_items(task_dir: Path, work: Path, v: dict, sp: dict, total: float, meta: dict,
                     look: dict, color: str):
    """Дубляж: окна с речью из своих видео -> перевод -> озвучка на тайм-кодах."""
    if v["kind"] != "own":
        raise RuntimeError("перевод речи возможен только для видеоряда «моё видео»")
    target = sp.get("target", "en")
    voice = dub.pick_voice(target, sp.get("voice"))
    n = int(v.get("clips", 1))
    if v.get("file"):
        pool, n = [_own_by_name(v["file"])], 1
    elif v.get("picks"):
        pool = _resolve_picks(v)  # выбранные вручную, в порядке выбора
        n = len(pool)
    else:
        pool = mix.own_clips()
        random.shuffle(pool)
    if not pool:
        raise RuntimeError("в _own нет видео: положите свои ролики в ShortsFactory/_clips/_own/ на Диске")
    fs, fe = v.get("start"), v.get("end")
    pool_all = mix.own_clips()
    items, groups, skipped, offset = [], [], [], 0.0
    for clip in pool:
        if len(items) >= n:
            break
        info = dub.analyze(clip)
        segs = info["segments"]
        if fs is not None:
            segs = [s for s in segs if s[0] >= fs - 0.1 and (fe is None or s[1] <= fe + 0.3)]
        if not segs:
            skipped.append(f"{clip.name}: нет речи" + (" в выбранном фрагменте" if fs is not None else ""))
            continue
        if info["lang"] == target:
            skipped.append(f"{clip.name}: уже на языке {target}")
            continue
        if fs is None and n == 1 and not v.get("seconds"):
            window = segs  # «целиком»: вся речь клипа
        elif fs is None:
            window = dub.pick_window(segs, total / n)
        else:
            window = segs
        w_start = fs if fs is not None else max(window[0][0] - 0.15, 0.0)
        rel = [[s - w_start, e - w_start, t] for s, e, t in window]
        texts = dub.translate([r[2] for r in rel], info["lang"], target)
        i = len(items)
        placed, voice_end = dub.build_voice(rel, texts, voice, work, f"c{i:02}", sp.get("rate") or None)
        want = (fe - fs) if (fs is not None and fe) else rel[-1][1] + 0.3
        length = min(max(want, voice_end + 0.15), video_duration(clip) - w_start)
        part = work / f"p{i:02}.mp4"
        layout = look.get("layout", "fill")
        clip2 = _second_clip(clip, pool_all) if layout == "split" else None
        _cut_styled(clip, w_start, length, part, layout=layout, clip2=clip2, color=color)  # скорость 1× - речь привязана к тайм-кодам
        real = video_duration(part)
        wav = work / f"v{i:02}.wav"
        dub.mix_voice_track(placed, real, wav)
        groups += dub.subtitle_groups(placed, texts, offset, real)
        item = {"part": part, "A": real, "voice": wav}
        if meta.get("_keep_orig"):
            orig = work / f"o{i:02}.wav"
            item["orig_real"] = _orig_audio(clip, w_start, real, orig)
            item["orig"] = orig
        items.append(item)
        library.touch(clip)
        offset += real
    dub.unload_translators()
    if not items:
        raise RuntimeError("нет подходящих видео с речью на другом языке. Пропущены: " + "; ".join(skipped[:6]))
    meta["skipped"] = skipped
    return items, groups


def _scene_items(work: Path, rec: dict, look: dict, color: str, meta: dict):
    """Сценарий из JSON: каждая сцена = фраза (озвучка) + свой клип; длина сцены = длина озвучки.
    Возвращает (items, groups): у каждого item есть готовая голосовая дорожка точной длины куска."""
    sp, v = rec["speech"], rec["video"]
    voice = sp.get("voice") or meta.get("voice")
    rate = sp.get("rate") or None
    keep = bool(v.get("orig_sound"))
    scenes = rec["scenes"]
    pool2 = _second_pool(v) if look.get("layout") == "split" else None
    warnings = meta.setdefault("warnings", [])
    items, groups, offset = [], [], 0.0
    for i, sc in enumerate(scenes):
        text = (sc.get("text") or "").strip()
        speech_len = 0.0
        if text:
            raw = work / f"sc{i:02}.mp3"
            tts.synthesize(text, raw, voice, rate)
            speech_len = audio_duration(raw)
            length = speech_len + (0.35 if i < len(scenes) - 1 else 0.7)
        else:
            length = float(sc.get("seconds") or 3)
        if sc.get("seconds"):
            length = max(length, float(sc["seconds"]))
        start = None
        if sc.get("own"):
            clip = _own_by_name(sc["own"], rec.get("project"))
            start = sc.get("start")
            library.touch(clip)
        else:
            theme = sc.get("theme") or v.get("theme") or None
            try:
                pool = mix._clips(theme)
            except RuntimeError as e:
                warnings.append(f"сцена {i + 1}: {str(e)[:120]}; взят любой клип")
                pool = mix._clips(None)
            clip = library.choose(pool)
        item = _fixed_parts(work, [(clip, start)], length, keep, look, pool2, color, n0=i)[0]
        wav = work / f"v{i:02}.wav"
        if text:
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-af", "apad", "-t", f"{item['A']:.3f}",
                 "-ar", "44100", "-ac", "1", str(wav)])
            groups += dub.subtitle_groups([(0.0, wav, speech_len)], [text], offset, item["A"])
        else:
            _silence(wav, item["A"])
        item["voice"] = wav
        item["sfx"] = sc.get("sfx")
        items.append(item)
        offset += item["A"]
    return items, groups


# ---------- финальная сборка ----------

def _xfade_chain(items: list[dict], style: dict) -> tuple[list[str], str, float]:
    """Фильтры склейки: xfade с переходами или обычный concat. Возвращает (фильтры, метка, длина)."""
    n = len(items)
    total = sum(i["A"] for i in items)
    d = style["d"] if style["transition"] and n > 1 else 0.0
    if n == 1:
        return ["[0:v]setpts=PTS-STARTPTS[c]"], "c", total
    if not d:
        labels = "".join(f"[{i}:v]" for i in range(n))
        return [f"{labels}concat=n={n}:v=1:a=0[c]"], "c", total
    f = [f"[{i}:v]tpad=stop_mode=clone:stop_duration={d},setpts=PTS-STARTPTS[p{i}]" for i in range(n)]
    cur, acc = "p0", items[0]["A"]
    for k in range(1, n):
        pool = style.get("transition_pool")
        tr = random.choice(pool) if pool else style["transition"]   # в каждой склейке свой переход из набора стиля
        f.append(f"[{cur}][p{k}]xfade=transition={tr}:duration={d}:offset={acc:.3f}[x{k}]")
        cur, acc = f"x{k}", acc + items[k]["A"]
    return f, cur, total


def _concat_wavs(task_dir: Path, wavs: list[Path], name: str) -> Path:
    lst = task_dir / f"{name}_list.txt"
    lst.write_text("".join(f"file '{w.resolve()}'\n" for w in wavs), "utf-8")
    out = task_dir / f"{name}.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-ar", "44100", "-ac", "1", str(out)])
    return out


def _logo_path(meta: dict) -> Path | None:
    found = sorted(config.ASSETS_DIR.glob(f"logo_{meta.get('channel', '')}.*"))
    return found[0] if found else None


def _overlays(meta: dict, look: dict, duration: float) -> list:
    """Текст поверх видео: хук в начале и призыв в конце. [(start, end, text, kind)]."""
    out = []
    hook, cta = look.get("hook", {}), look.get("cta", {})
    text = ""
    if hook.get("mode") == "random":
        text = random.choice(HOOK_PRESETS)
    elif hook.get("mode") == "title":
        title = (meta.get("title") or "").strip()
        text = "" if title in ("", "Ролик") or title.startswith("Нарезка") else title
    elif hook.get("mode") == "text":
        text = (hook.get("text") or "").strip()
    if text:
        text = text if len(text) <= 80 else text[:77] + "..."
        start = min(max(float(hook.get("start", 0.15)), 0.0), max(duration - 1.0, 0.0))
        dur = hook.get("dur", "auto")
        span = min(3.4, max(1.6, duration * 0.35)) if dur in (None, "auto") else float(dur)
        out.append((start, min(start + span, duration), text, "hook"))
    if cta.get("mode") == "text" and (cta.get("text") or "").strip():
        d = cta.get("dur", "auto")
        span = min(3.6, max(1.5, duration * 0.3)) if d in (None, "auto") else min(float(d), duration)
        when = cta.get("when", "end")
        start = 0.5 if when == "start" else (max((duration - span) / 2, 0.0) if when == "mid" else max(duration - span, 0.0))
        out.append((start, min(start + span, duration), cta["text"].strip()[:80], "cta"))
    return out


def _pick_sfx(tag: str | None, prefer: tuple[str, ...]) -> Path | None:
    """Звук из библиотеки: явный тег выбора важнее, иначе предпочтительные теги, иначе любой."""
    for wanted in ([tag] if tag else [*prefer, None]):
        rows = library.search("sfx", [wanted] if wanted else [], limit=200)
        files = [config.ROOT / r["path"] for r in rows if (config.ROOT / r["path"]).exists()]
        if files:
            return library.choose(files)
    return None


def _sfx_events(items: list[dict], style: dict, sfxc: dict, duration: float) -> list[tuple[float, Path]]:
    """[(время, файл)]: звук в начале и/или на стыках клипов. У сцены сценария может быть свой тег звука
    (item['sfx']: строка - этот тег, False - без звука на этом стыке)."""
    events = []
    tag = sfxc.get("tag")
    lead = 0.12 if style.get("transition") else 0.02  # свист начинается чуть раньше перехода
    starts, acc = [0.0], 0.0
    for it in items[:-1]:
        acc += it["A"]
        starts.append(acc)
    for k, t in enumerate(starts):
        if k > 0 and t >= duration - 0.3:
            continue
        per = items[k].get("sfx")
        when = 0.0 if k == 0 else max(t - lead, 0.0)
        if per is False:
            continue
        if isinstance(per, str) and per:
            f = _pick_sfx(per, ())
        elif k == 0 and sfxc.get("start"):
            f = _pick_sfx(tag, ("hit", "pop"))
        elif k > 0 and sfxc.get("cuts"):
            f = _pick_sfx(tag, ("whoosh", "click"))
        else:
            continue
        if f:
            events.append((when, f))
    return events


def assemble(task_dir: Path, items: list[dict], audio: Path | None, groups: list, meta: dict, style: dict,
             look: dict, sfxc: dict) -> Path:
    vrec = (meta.get("recipe") or {}).get("video") or {}
    speech = audio
    if speech is None and all(i.get("voice") for i in items):
        speech = _concat_wavs(task_dir, [i["voice"] for i in items], "speech_all")
    orig = _concat_wavs(task_dir, [i["orig"] for i in items], "orig_all") if all(i.get("orig") for i in items) else None
    warnings = meta.setdefault("warnings", [])
    if meta.get("_subs_orig") and not groups and orig is not None:
        # субтитры по звуку видео: Whisper слушает речь самих клипов (язык определяется сам)
        if not any(i.get("orig_real") for i in items):
            warnings.append("субтитры по звуку видео не созданы: у клипов нет звуковой дорожки")
        else:
            words = subtitles.transcribe_words(orig, "auto")
            groups = subtitles.group_words(words) if words else []
            if not groups:
                warnings.append("в звуке видео не распознана речь - субтитров нет")
    if meta.get("_orig_muted"):
        orig_for_mix = None  # звук оригинала нужен был только для субтитров
    else:
        orig_for_mix = orig

    filters, cur, duration = _xfade_chain(items, style)
    post = []
    if style.get("grade") in GRADES:
        post.append(GRADES[style["grade"]])
    zf = montage.zoom_filter(style.get("zoom"))
    if zf:
        post.append(zf)
    fx_all = list(dict.fromkeys([*style.get("fx", []), *look.get("fx", [])]))   # эффекты стиля + выбранные отдельно
    post += [FX_FILTERS[f] for f in fx_all if f in FX_FILTERS]
    want_subs = bool(groups) and meta.get("subs", True)
    overlays = _overlays(meta, look, duration)
    if want_subs or overlays:
        subtitles.write_ass(groups if want_subs else [], task_dir / "subs.ass", look, overlays)
        post.append("subtitles=subs.ass")

    logo = _logo_path(meta) if look.get("logo", "off") != "off" else None
    if look.get("logo", "off") != "off" and logo is None:
        warnings.append(f"логотип не загружен для канала «{meta.get('channel')}» - пропущен")
    chain = ",".join(post) if post else "null"
    n = len(items)
    if logo:
        filters.append(f"[{cur}]{chain}[vp]")
        filters.append(f"[{n}:v]scale=220:-1,format=rgba,colorchannelmixer=aa=0.88[lg]")
        filters.append(f"[vp][lg]overlay={LOGO_XY.get(look['logo'], LOGO_XY['tr'])}:format=auto[v]")
    else:
        filters.append(f"[{cur}]{chain}[v]")

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for it in items:
        cmd += ["-i", str(it["part"])]
    idx = n
    if logo:
        cmd += ["-i", str(logo)]
        idx += 1
    music = mix._music(meta.get("music", "off"))
    # (путь, громкость, опции входа, название): речь 100%, оригинал 100% если он один, иначе 10%, музыка - по настройке
    tracks = []
    if speech is not None:
        tracks.append((speech, 1.0, [], "речь"))
    if orig_for_mix is not None:
        ov = vrec.get("orig_vol")
        tracks.append((orig_for_mix, (int(ov) / 100 if ov else (0.10 if speech is not None else 1.0)), [], "оригинал"))
    if music is not None:
        vol = int(meta.get("music_vol", 12)) / 100 if tracks else 1.0
        tracks.append((music, vol, ["-stream_loop", "-1"], "музыка"))
    labels = []
    names = [t[3] for t in tracks]
    fade = max(0.0, duration - 1.5)
    for k, (path, vol, opts, name) in enumerate(tracks):
        cmd += [*opts, "-i", str(path)]
        tail = f",afade=t=out:st={fade:.2f}:d=1.5" if name == "музыка" else ""
        filters.append(f"[{idx}:a]aresample=44100,aformat=channel_layouts=mono,volume={vol:.2f}{tail}[s{k}]")
        labels.append(f"[s{k}]")
        idx += 1
    if vrec.get("duck", True) and "оригинал" in names and "музыка" in names:
        # музыка автоматически тише, пока в клипе звучит оригинал (лай, голос), и возвращается в паузах
        o_i, m_i = names.index("оригинал"), names.index("музыка")
        filters.append(f"{labels[o_i]}asplit=2[so1][so2]")
        filters.append(f"{labels[m_i]}[so2]sidechaincompress=threshold=0.03:ratio=8:attack=15:release=600[smd]")
        labels[o_i], labels[m_i] = "[so1]", "[smd]"
    events = _sfx_events(items, style, sfxc, duration)
    if (sfxc.get("cuts") or sfxc.get("start")) and not events:
        warnings.append("звуковые эффекты выбраны, но в библиотеке нет подходящих звуков")
    sfx_vol = int(sfxc.get("vol", 60)) / 100
    for k, (t, path) in enumerate(events):
        cmd += ["-i", str(path)]
        filters.append(
            f"[{idx}:a]atrim=0:1.8,asetpts=PTS-STARTPTS,aresample=44100,aformat=channel_layouts=mono,"
            f"afade=t=out:st=1.4:d=0.4,volume={sfx_vol:.2f},adelay={int(t * 1000)}:all=1[e{k}]")
        labels.append(f"[e{k}]")
        idx += 1
    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[am]")
    elif labels:
        filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:dropout_transition=0:normalize=0[am]")
    if labels:
        filters.append(f"[am]{LOUDNESS}[a]")
    cmd += ["-filter_complex", ";".join(filters), "-map", "[v]"]
    cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", random.choice(["176k", "192k", "224k"]), "-ar", "44100"] if labels else ["-an"]
    out = task_dir / "final.mp4"
    cmd += ["-t", f"{duration:.2f}", "-c:v", "libx264", "-preset", "fast", *_encode_variation(), "-maxrate", MAXRATE,
            "-bufsize", "24M", "-profile:v", "high", "-pix_fmt", "yuv420p", *ANON_FLAGS, "-movflags", "+faststart", str(out)]
    run(cmd, cwd=task_dir)
    scrub(out)
    meta["audio_report"] = {
        "речь": speech is not None, "оригинал": orig_for_mix is not None,
        "музыка": music.name if music else None,
        "звуки": [f"{t:.1f}с {p.name}" for t, p in events],
        "дорожки": [t[3] for t in tracks] + (["эффекты"] if events else []) or "звука нет",
    }
    if orig_for_mix is not None and not any(i.get("orig_real") for i in items):
        meta["audio_report"]["предупреждение_оригинал"] = "у выбранных видео нет звуковой дорожки, оригинал беззвучный"
    if meta.get("music", "off") != "off" and music is None:
        meta["audio_report"]["предупреждение"] = f"музыка «{meta.get('music')}» не найдена в библиотеке (проверьте _clips/_music/)"
    log.info("звук: %s", meta["audio_report"])
    meta["music_used"] = music.name if music else None
    meta["duration"] = round(duration, 1)
    if not warnings:
        meta.pop("warnings", None)
    return out


def build(task_dir: Path, meta: dict) -> Path:
    rec = meta["recipe"]
    v, sp = rec["video"], rec["speech"]
    style = resolve_style(rec.get("style", "none"))
    look, sfxc = merged_look(rec.get("look")), merged_sfx(rec.get("sfx"))
    if sp["kind"] == "translate":
        look["speed"] = "1"  # озвучка привязана к тайм-кодам оригинала
    color = frame_spec(look)
    meta["style_used"] = {k: style.get(k) for k in ("name", "transition", "transition_pool", "d", "grade", "zoom", "fx")}
    meta["look_used"] = {**look, **({"frame_used": color} if look["layout"] == "frame" else {})}

    mix.sync_clips()
    library.scan()
    work = task_dir / "parts"
    work.mkdir(exist_ok=True)

    audio, groups, kind = None, [], sp["kind"]
    total = float(v.get("seconds") or 30)
    # звук оригинала: None = авто (включён, если речи нет и видеоряд - клипы или моё видео)
    orig_pref = v.get("orig_sound")
    keep_orig = (kind == "none" and v["kind"] in ("clips", "own")) if orig_pref is None else bool(orig_pref) and v["kind"] != "bg"
    # субтитры по звуку видео: оригинал извлекается всегда, а в микс идёт только если он не выключен
    subs_orig = bool(meta.get("subs", True)) and kind == "none" and rec.get("subs_src") == "orig" and v["kind"] != "bg"
    meta["_subs_orig"] = subs_orig
    meta["_orig_muted"] = subs_orig and not keep_orig
    keep_orig = keep_orig or subs_orig
    meta["_keep_orig"] = keep_orig

    if kind == "translate":
        items, groups = _translate_items(task_dir, work, v, sp, total, meta, look, color)
    elif kind == "scenes":
        items, groups = _scene_items(work, rec, look, color, meta)
    else:
        if kind == "audio":
            audio = next((f for f in task_dir.iterdir() if f.name.startswith("input_audio")), None)
            if audio is None:
                raise RuntimeError("в папке задачи нет аудио")
            total = audio_duration(audio)
        elif kind == "text":
            audio = task_dir / "voice.mp3"
            voice = sp.get("voice") or meta.get("voice")
            tts.synthesize((task_dir / "script.txt").read_text("utf-8"), audio, voice, sp.get("rate") or None)
            total = audio_duration(audio)
        elif v["kind"] == "own" and v.get("file") and v.get("start") is None:
            total = video_duration(_own_by_name(v["file"]))  # своё видео без речи «целиком»
        items = _video_items(work, v, total, keep_orig, look, color, exact=kind in ("audio", "text"))
        if audio is not None and meta.get("subs", True):
            lang = (sp.get("voice") or meta.get("voice") or "")[:2] if kind == "text" else meta.get("lang")
            words = subtitles.transcribe_words(audio, lang)
            groups = subtitles.group_words(words) if words else []

    out = assemble(task_dir, items, audio, groups, meta, style, look, sfxc)
    shutil.rmtree(work, ignore_errors=True)
    for k in ("_keep_orig", "_subs_orig", "_orig_muted"):
        meta.pop(k, None)
    for junk in ("speech_all_list.txt", "speech_all.wav", "orig_all_list.txt", "orig_all.wav", "voice.mp3", "script.txt", "subs.ass"):
        (task_dir / junk).unlink(missing_ok=True)
    for f in list(task_dir.glob("input_audio*")):
        f.unlink(missing_ok=True)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("ffmpeg не создал final.mp4")
    return out

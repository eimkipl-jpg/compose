#!/usr/bin/env python3
"""Сборщик патчей для бота: сравнивает две папки проекта и делает один файл-установщик.

Запуск (на своём компьютере или там, где лежат обе папки):
    python make_patch.py <старая_папка> <новая_папка> <имя_патча.py> [описание]
Старая папка - копия проекта ДО ваших правок, новая - ПОСЛЕ. Остальное делается само:
- в патч попадают только изменённые, новые и удалённые файлы (кэш, .env, базы и логи пропускаются);
- на сервере установщик сначала делает пробный прогон и ничего не трогает, если правки не встают;
- перед изменением файлы копируются в ~/shorts_factory_backup_<дата>/;
- повторный запуск безопасен: уже применённый патч распознаётся.
Установка на сервере: cd ~ && python3 <имя_патча>.py && systemctl restart shorts-gateway shorts-worker
"""
import base64, difflib, gzip, shutil, subprocess, sys
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", "tasks", "tmp", "logs"}
SKIP_FILES = {".env", "factory.db"}
SKIP_SUFFIX = {".pyc", ".db", ".sqlite", ".log", ".mp4", ".mp3", ".wav", ".mov"}


def files_of(root: Path) -> dict[str, Path]:
    out = {}
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if not p.is_file() or set(rel.parts) & SKIP_DIRS or p.name in SKIP_FILES or p.suffix in SKIP_SUFFIX:
            continue
        out[rel.as_posix()] = p
    return out


def make_diff(old: Path, new: Path) -> str:
    a, b = files_of(old), files_of(new)
    chunks = []
    for rel in sorted(set(a) | set(b)):
        x = a[rel].read_text("utf-8").splitlines(keepends=True) if rel in a else []
        y = b[rel].read_text("utf-8").splitlines(keepends=True) if rel in b else []
        if x == y:
            continue
        chunks.append("".join(difflib.unified_diff(x, y, f"a/{rel}" if rel in a else "/dev/null",
                                                   f"b/{rel}" if rel in b else "/dev/null", n=3)))
    return "".join(chunks)


INSTALLER = '''"""{doc}
Запуск на сервере: cd ~ && python3 {name} && systemctl restart shorts-gateway shorts-worker
Резервная копия: ~/shorts_factory_backup_<дата>/
"""
import base64, gzip, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/shorts_factory")
if not TARGET.exists():
    sys.exit(f"Не найдена папка {{TARGET}}")
if not shutil.which("patch"):
    sys.exit("Нет утилиты patch. Установите: apt install -y patch")
DIFF = gzip.decompress(base64.b64decode("".join(DATA.split()))).decode("utf-8")
tmp = Path(tempfile.mkdtemp()) / "update.diff"
tmp.write_text(DIFF, "utf-8")
files = sorted(set(re.findall(r"^\\+\\+\\+ b/(\\S+)", DIFF, re.M)) | set(re.findall(r"^--- a/(\\S+)", DIFF, re.M)))


def patch(*flags):
    return subprocess.run(["patch", "-p1", "-d", str(TARGET), "-i", str(tmp), *flags], capture_output=True, text=True)


if patch("-R", "--dry-run", "--silent").returncode == 0:
    sys.exit("Этот патч уже установлен, ничего не делаю.")
dry = patch("--dry-run")
if dry.returncode != 0:
    print("ОСТАНОВЛЕНО, файлы не тронуты. Пришлите этот вывод:")
    print(dry.stdout[-1500:], dry.stderr[-500:])
    sys.exit(1)
backup = Path.home() / f"shorts_factory_backup_{{time.strftime('%Y%m%d_%H%M%S')}}"
for rel in files:
    src = TARGET / rel
    if src.exists():
        (backup / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, backup / rel)
res = patch("--no-backup-if-mismatch")
print(res.stdout.strip())
if res.returncode != 0:
    sys.exit("Патч не применился полностью, файлы можно вернуть из " + str(backup))
print(f"Готово: изменено файлов {{len(files)}}. Резервная копия: {{backup}}")
print("Теперь: systemctl restart shorts-gateway shorts-worker")
'''


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    old, new, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    doc = sys.argv[4] if len(sys.argv) > 4 else "Патч бота."
    diff = make_diff(old, new)
    if not diff:
        sys.exit("Различий между папками нет, патч не нужен.")
    data = base64.b64encode(gzip.compress(diff.encode("utf-8"))).decode()
    data = "\n".join(data[i:i + 120] for i in range(0, len(data), 120))
    code = INSTALLER.format(doc=doc, name=out.name) + f'\nDATA = """\n{data}\n"""\n'
    # DATA должна быть определена до использования: переносим в начало после импортов
    head, _, rest = code.partition("TARGET = ")
    code = head + f'DATA = """\n{data}\n"""\nTARGET = ' + rest.replace(f'\nDATA = """\n{data}\n"""\n', "\n")
    out.write_text(code, "utf-8")
    n = len({l[6:] for l in diff.splitlines() if l.startswith("+++ b/")} | {l[6:] for l in diff.splitlines() if l.startswith("--- a/")})
    print(f"Готово: {out} (файлов в патче: {n})")


main()

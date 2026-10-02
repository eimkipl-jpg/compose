"""Что меняет патч
Запуск на сервере: cd ~ && python3 my_patch.py && systemctl restart shorts-gateway shorts-worker
Резервная копия: ~/shorts_factory_backup_<дата>/
"""
import base64, gzip, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

DATA = """
H4sIAEt9v2oC/7VWT2/jRBS/51M8mUOSzTh1stk2MTLqAcQF7SKtkEBWFE2dSTLU9njtSXZLiARaAQcOVIjCaoGygOACEhVQtcuy32HyjXjjcdw23RYu5GB7
3p/fe35/fo5t20A3ApEyvESJyFgz2as0Gg3YuSTd3gZ7y+mQTWjoW2sTtrcrMGQjoFnGop2Q1STNdgdDnrrwJpUTAlyyKHMh5Jn0hzyQfQJ0OuTC6OF9uC1i
RmCcimlS2BGImKQuaHMCmdwLWXGowIVfKMRuaTZ6EJjnOtiv5OCuMZ+lLAAPahq0OWayZqGAJ8yqg0hhvqgb4YwPmVjJKg3taa4vgY0lUt+oR+qxeqIObXz8
Uf2g9tXPKHrkgvpDHavn6nj5gTpafqxOl5/B8kP1bPlw+Yk6Uk/V8+Wny49AnaDNEYq07al6CupPtEHTh/n5LzR9hvcTg6FOdVSTQJCKZHAfXyGl8VBETX3j
say1CWzWz5lMrjVJ6FDrxTQe1gqracxHIo1qTtO5RcBp3qwTuFmY76R8PJFXedjo0tYuTvucT0bl1SF6XQKti/ax4Bk7SzqYCB6wmt8i0O6vsuYymFwFinBO
K0d1OgjbKXxM27OEsdw1Hzcj46OVmGf54AECAg3DGi+GQGeAQ4DwwIHHZnzr7tnglbCDQMQBlYP7dJaVU0/A536B0l9Dwcm3jPcAI1p1s02dTdLDbbq1SVq9
fJuKPEMxFufCjngoWZo1aZIwLMTI8ufBNF3058GE8njhz5I+Il5nHi/cWT8LaMi8dttx7RbRNaTSS8c7lAQiFClixTELI/6ApR6lntPsdv1wrJHtq5ExtDYS
M5aGdM+bv3Hn9TuDt9/JC6o31K/qd6ni6xcavyrTar++cIsE6FQKf6ajNP73KIMkZWWlWJgx97pXW6vxv6S4Zr0KdZlKPkfe+Ao5ZF99rX5V36pf1AGsk0q+
GyRfbCQ33TYC7B4Bqfe4pIa1JM5lZxUJ5DgeDlzmzfPnhYuHkSeJhvbmLafrgA1tuFHwzMKdt3rtC7LJgljnoc0UaU9XmxJ2zzN0EbMMw5jnhYt0ME2p5CL2
5vi8BqJfxMskwkdiyLwgxHV08/Ow9EKTvOzG70WlfKwOsXLfqQMs4/dF9X4D9ZMmahRp5QGW9QmqvlT7rmbb35GBkWmRiU/x+QhPf8MqJGiOhgslLlUNTysq
Bb0EkaZT3xqNooSNLdxtey+/4hiGbMZCfWBpKlKrbzxyNpAlHbhm/3s9sgWNbqtHuuX6s1ATAN3B+byOAubVavNdweOaMa0vKG6ux+NkKrELITtTuGVJscpj
lkl3iH0VUzmQSKkZz1WOG+tNCfl7zHN8GpVrci6Za3YFHXAp33r19mt37y58emFXimbt44h/gQ06vDTthub1xxE/osfqBP9PMExNsg2asoxGScjOOvIf4pOV
v9fpOo5zY54HQPEKzchNmmU/G7qhtoEe6L89IXug2/iyZepcBK3rNkc00So9nf11hJUO0bVp4FJ9pzTIjzv5ce2bZ7W2Nne1Ghcqv7fbnV2rn4eiqRZ0Oi3H
wY9K2Y2cvnQ4Gq9SwI7iUK4+RrAB1ojHNGxGSceq/AO88gKb6QkAAA==
"""
TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/shorts_factory")
if not TARGET.exists():
    sys.exit(f"Не найдена папка {TARGET}")
if not shutil.which("patch"):
    sys.exit("Нет утилиты patch. Установите: apt install -y patch")
DIFF = gzip.decompress(base64.b64decode("".join(DATA.split()))).decode("utf-8")
tmp = Path(tempfile.mkdtemp()) / "update.diff"
tmp.write_text(DIFF, "utf-8")
files = sorted(set(re.findall(r"^\+\+\+ b/(\S+)", DIFF, re.M)) | set(re.findall(r"^--- a/(\S+)", DIFF, re.M)))


def patch(*flags):
    return subprocess.run(["patch", "-p1", "-d", str(TARGET), "-i", str(tmp), *flags], capture_output=True, text=True)


if patch("-R", "--dry-run", "--silent").returncode == 0:
    sys.exit("Этот патч уже установлен, ничего не делаю.")
dry = patch("--dry-run")
if dry.returncode != 0:
    print("ОСТАНОВЛЕНО, файлы не тронуты. Пришлите этот вывод:")
    print(dry.stdout[-1500:], dry.stderr[-500:])
    sys.exit(1)
backup = Path.home() / f"shorts_factory_backup_{time.strftime('%Y%m%d_%H%M%S')}"
for rel in files:
    src = TARGET / rel
    if src.exists():
        (backup / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, backup / rel)
res = patch("--no-backup-if-mismatch")
print(res.stdout.strip())
if res.returncode != 0:
    sys.exit("Патч не применился полностью, файлы можно вернуть из " + str(backup))
print(f"Готово: изменено файлов {len(files)}. Резервная копия: {backup}")
print("Теперь: systemctl restart shorts-gateway shorts-worker")


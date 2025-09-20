from pathlib import Path
import re

ALLOW = re.compile(r'^[\u0900-\u097F\s]+$')  # Devanagari + whitespace
FILES = [
    "data/S1_clean/train_corpus.txt",
    "data/S1_clean/val_corpus.txt",
]

for fp in FILES:
    p = Path(fp)
    if not p.exists():
        print(f"Missing: {fp}")
        continue
    lines = p.read_text(encoding="utf-8-sig").splitlines()  # auto-strip BOM
    kept, dropped = [], []
    for ln in lines:
        s = ln.strip().lstrip("\ufeff")
        if not s:
            continue
        (kept if ALLOW.match(s) else dropped).append(s)
    p.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"{fp}: kept {len(kept)} lines, dropped {len(dropped)} bad lines")
    if dropped:
        print("  e.g. dropped:", dropped[:3])

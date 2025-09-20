# src/audit_chars.py
import json
from pathlib import Path

LABELS = "labels/classes_vedic.json"
SPLITS = [
    "data/S1_clean/train/transcripts.txt",
    "data/S1_clean/val/transcripts.txt",
]

classes = set(json.loads(Path(LABELS).read_text(encoding="utf-8")))
seen = set()

for p in SPLITS:
    t = Path(p).read_text(encoding="utf-8").splitlines()
    for ln in t:
        if not ln.strip(): continue
        try:
            _, text = ln.split("\t", 1)
        except ValueError:
            continue
        for ch in text:
            seen.add(ch)

unknown = sorted(ch for ch in seen if ch not in classes and ch != "\t")
print("UNKNOWN CHARS:")
for ch in unknown:
    print(f"{ch}  U+{ord(ch):04X}")

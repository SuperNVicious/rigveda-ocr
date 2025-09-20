
import argparse, json, re, unicodedata
from collections import Counter
from typing import Dict, Tuple, List

def nfc_norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s\u0900-\u097F\u0951\u0952]", " ", s)  # keep Devanagari + Vedic marks

def toks(s: str) -> List[str]:
    s = strip_punct(nfc_norm(s)).lower()
    return [t for t in s.split() if t]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def edit_ratio(a: str, b: str) -> float:
    try:
        import editdistance
        d = editdistance.eval(a, b)
        n = max(1, len(a))
        return 1 - d/n
    except Exception:
        return 0.0

def load_keylex(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def keyterm_recall(ocr_sa: str, tr_en: str, keylex: List[Dict]) -> Tuple[float, List[str], List[str]]:
    # very light matching: if any english synonym appears in translation, we consider the Sanskrit term covered
    tr_toks = set(toks(tr_en))
    matched, missing = [], []
    extras = []
    for entry in keylex:
        sa = entry["sa"]
        en_syns = [t.lower() for t in entry["en"]]
        ok = any(any(sub in " ".join(tr_toks) for sub in syn.split()) for syn in en_syns)
        if ok:
            matched.append(sa)
        else:
            # only mark missing if the Sanskrit term actually appears in OCR
            if sa in ocr_sa:
                missing.append(sa)
    # extra: translation terms that are alien (very simple heuristic: proper nouns not in any syn list)
    all_syns_flat = set(sum(([w.lower() for w in entry["en"]] for entry in keylex), []))
    extras = [t for t in tr_toks if t not in all_syns_flat and t.isalpha() and len(t)>5][:5]
    recall = 0.0 if (matched or missing) == [] else (len(matched) / max(1, (len(matched)+len(missing))))
    return recall, matched, missing

def semantic_sim(ocr_sa: str, tr_en: str) -> float:
    # Placeholder: if sentence-transformers is available locally, plug it here.
    # For now we approximate via token overlap on transliterated-ish Latin fallback.
    return jaccard(toks(ocr_sa), toks(tr_en))

def fidelity_score(ocr_sa: str, tr_en: str, keylex: List[Dict]) -> Dict:
    o_sa = nfc_norm(ocr_sa)
    t_en = nfc_norm(tr_en)

    tok_overlap = jaccard(toks(o_sa), toks(t_en))
    edit = edit_ratio(o_sa, t_en)
    krec, kmatch, kmiss = keyterm_recall(o_sa, t_en, keylex)
    sem = semantic_sim(o_sa, t_en)

    score = round(100 * (0.50*sem + 0.20*krec + 0.15*tok_overlap + 0.15*edit), 1)
    verdict = "Essence preserved" if score>=85 else ("Mostly preserved" if score>=70 else ("Partial" if score>=50 else "Low fidelity"))

    return {
        "score": score,
        "subscores": {
            "SemanticSim": round(sem,3),
            "KeyTermRecall": round(krec,3),
            "TokenOverlap": round(tok_overlap,3),
            "EditRatio": round(edit,3)
        },
        "matched_terms": kmatch,
        "missing_terms": kmiss,
        "verdict": verdict
    }

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr_text", required=True)
    ap.add_argument("--translation", required=True)
    ap.add_argument("--keylex", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    keylex = load_keylex(args.keylex)
    res = fidelity_score(args.ocr_text, args.translation, keylex)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    cli()

# tools/inspect_val_preds.py
import torch
from src.tokenizer import Tokenizer
from src.crnn_model import CRNN
from src.dataset import LineDataset

def ctc_greedy_collapse(argmax_ids, blank_idx: int):
    """
    argmax_ids: List[int] for one sample (length T)
    Collapse rule: remove blanks; merge repeats relative to last *emitted* symbol.
    """
    out = []
    for s in argmax_ids:
        if s == blank_idx:
            continue
        if not out or s != out[-1]:
            out.append(s)
    return out

def main():
    ckpt = "runs/crnn/rigveda_s1_crnn_best.pt"
    labels = "labels/classes_vedic.json"
    val_root = "data/S1_clean/val"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = Tokenizer(labels)
    model = CRNN(num_classes=len(tok.tokens)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ds = LineDataset(val_root)
    print(f"Loaded {len(ds)} validation samples\n")

    for i in range(min(10, len(ds))):
        img, gt = ds[i]
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            lp = model(x)              # (1, T, C) log-probs
        ids_t = lp[0].argmax(dim=-1).cpu().tolist()   # raw argmax over time
        ids = ctc_greedy_collapse(ids_t, blank_idx=tok.blank_index)
        pred = tok.ids_to_text(ids)
        print(f"{i+1:02d}) GT: {gt}\n    PD: {pred}\n")

if __name__ == "__main__":
    main()

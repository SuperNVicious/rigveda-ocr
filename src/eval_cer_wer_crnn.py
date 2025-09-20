import argparse, torch
from tqdm import tqdm
import editdistance as ed
from .dataset import LineDataset
from .tokenizer import Tokenizer
from .crnn_model import CRNN

def ctc_greedy_decode(log_probs: torch.Tensor, blank_idx: int = 0):
    seq = log_probs.argmax(dim=-1).squeeze(0).tolist()
    out, prev = [], None
    for s in seq:
        if s != blank_idx and s != prev:
            out.append(s)
        prev = s
    return out

def cer(a: str, b: str) -> float:
    if not a and not b: return 0.0
    return ed.eval(list(a), list(b)) / max(1, len(b))

def wer(a: str, b: str) -> float:
    A, B = a.split(), b.split()
    if not A and not B: return 0.0
    return ed.eval(A, B) / max(1, len(B))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='path to val split folder containing transcripts.txt')
    ap.add_argument('--labels', required=True, help='labels/classes_vedic.json')
    ap.add_argument('--ckpt', required=True, help='checkpoint .pt path')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Tokenizer(args.labels)
    ds = LineDataset(args.data_root)
    model = CRNN(num_classes=len(tokenizer.classes)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    tot_cer = 0.0
    tot_wer = 0.0
    n = 0

    for img, gt in tqdm(ds, desc='Evaluating'):
        x = img[None, :, :, :].to(device)
        with torch.no_grad():
            log_probs = model(x)  # (1,T,C)
        ids = ctc_greedy_decode(log_probs.cpu(), blank_idx=tokenizer.blank_index)
        pred = tokenizer.ids_to_text(ids)
        tot_cer += cer(pred, gt)
        tot_wer += wer(pred, gt)
        n += 1

    print(f"Avg CER: {tot_cer/max(1,n):.4f} | Avg WER: {tot_wer/max(1,n):.4f}")

if __name__ == '__main__':
    main()

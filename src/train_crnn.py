import argparse, os, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import editdistance as ed

from .dataset import LineDataset
from .tokenizer import Tokenizer
from .crnn_model import CRNN

# -------------------------
# Helpers
# -------------------------
def ctc_greedy_decode(log_probs: torch.Tensor, blank_idx: int = 0):
    seq = log_probs.argmax(dim=-1).tolist()
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

def collate(batch, tokenizer: Tokenizer):
    imgs, texts = zip(*batch)
    imgs = torch.stack(imgs)
    targets = [torch.tensor(tokenizer.text_to_ids(t), dtype=torch.long) for t in texts]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets) if len(targets) > 0 else torch.empty(0, dtype=torch.long)
    return imgs, targets_concat, target_lengths, list(texts)

def evaluate(loader, model, tokenizer, device):
    model.eval()
    tot_cer, tot_wer, n = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, _, _, texts in loader:
            imgs = imgs.to(device, non_blocking=True)
            log_probs = model(imgs)
            for bi in range(imgs.size(0)):
                ids = ctc_greedy_decode(log_probs[bi].cpu(), blank_idx=tokenizer.blank_index)
                pred = tokenizer.ids_to_text(ids)
                gt = texts[bi]
                tot_cer += cer(pred, gt)
                tot_wer += wer(pred, gt)
                n += 1
    return (tot_cer / max(1, n), tot_wer / max(1, n))

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Tokenizer(args.labels)

    train_ds = LineDataset(os.path.join(args.data_root, 'train'))
    val_ds   = LineDataset(os.path.join(args.data_root, 'val'))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                              collate_fn=lambda b: collate(b, tokenizer))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                              collate_fn=lambda b: collate(b, tokenizer))

    model = CRNN(num_classes=len(tokenizer.classes)).to(device)
    ctc_loss = nn.CTCLoss(blank=tokenizer.blank_index, zero_infinity=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_cer = float('inf')
    best_path = args.out.replace('.pt', '_best.pt')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, targets_concat, target_lengths, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets_concat = targets_concat.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                log_probs = model(imgs)
                T = log_probs.size(1)
                input_lengths = torch.full((log_probs.size(0),), T, dtype=torch.long, device=device)
                loss = ctc_loss(log_probs.permute(1, 0, 2), targets_concat, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        train_cer, train_wer = evaluate(train_loader, model, tokenizer, device)
        val_cer, val_wer = evaluate(val_loader, model, tokenizer, device)

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Train CER: {train_cer:.3f}, WER: {train_wer:.3f} "
              f"| Val CER: {val_cer:.3f}, WER: {val_wer:.3f}")

        # Save best model
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), best_path)
            print(f"  >> New best model saved at {best_path} (Val CER {val_cer:.3f})")

    print(f"Training complete. Best model at {best_path} with CER {best_cer:.3f}")

if __name__ == '__main__':
    main()

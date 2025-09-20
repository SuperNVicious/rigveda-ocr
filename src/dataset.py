from pathlib import Path
import torch, cv2
from torch.utils.data import Dataset
from .preproc import letterbox
from .tokenizer import Tokenizer

class LineDataset(Dataset):
    def __init__(self, split_dir: str):
        self.split_dir = Path(split_dir)
        tfile = self.split_dir / 'transcripts.txt'
        lines = tfile.read_text(encoding='utf-8-sig').splitlines()
        self.transcripts = []
        for ln in lines:
            if not ln.strip():
                continue
            path_rel, text = ln.split('\t', 1)
            self.transcripts.append((path_rel, text))

        # hard guard: fail fast if unknown chars exist
        tok = Tokenizer("labels/classes_vedic.json")
        allowed = set(tok.token_to_id.keys())
        for _, text in self.transcripts:
            bad = [ch for ch in text if ch not in allowed]
            if bad:
                raise ValueError(f"Unknown chars in dataset: {bad[:5]}...  Update labels/classes_vedic.json")

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx: int):
        path_rel, text = self.transcripts[idx]
        img_path = self.split_dir / path_rel
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            import numpy as np
            img = (255*np.ones((48,512), dtype='uint8'))
        img = letterbox(img, 48, 512)
        img = img[None, :, :]
        return torch.tensor(img, dtype=torch.float32), text

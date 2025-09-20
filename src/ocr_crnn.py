from typing import Optional
import torch, cv2
from .tokenizer import Tokenizer
from .crnn_model import CRNN
from .preproc import letterbox

def ctc_greedy_decode(log_probs: torch.Tensor, blank_idx: int = 0):
    """
    log_probs: (1, T, C) torch.Tensor
    returns: list[int] (collapsed, no blanks)
    """
    seq = log_probs.argmax(dim=-1).squeeze(0).tolist()
    out = []
    prev = None
    for s in seq:
        if s != blank_idx and s != prev:
            out.append(s)
        prev = s
    return out

class RigvedaCRNN:
    def __init__(self, ckpt: str, classes_path: str, device: Optional[str] = None):
        self.tokenizer = Tokenizer(classes_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CRNN(num_classes=len(self.tokenizer.classes)).to(self.device)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

    def infer_image(self, img_path: str) -> str:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = letterbox(img, 48, 512)
        x = torch.tensor(img[None, None, ...], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            log_probs = self.model(x)  # (1, T, C)
        ids = ctc_greedy_decode(log_probs.cpu(), blank_idx=self.tokenizer.blank_index)
        return self.tokenizer.ids_to_text(ids)

import os, sys, tempfile, shutil, torch, gradio as gr
from PIL import Image

# ensure local package imports work when running with -m
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(__file__))

from src.tokenizer import Tokenizer
from src.crnn_model import CRNN
from src.dataset import LineDataset

CKPT   = os.path.join("runs","crnn","rigveda_s1_crnn_best.pt")
LABELS = os.path.join("labels","classes_vedic.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer(LABELS)
model = CRNN(num_classes=len(tok.tokens)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

def ctc_collapse(ids, blank_idx):
    out=[]
    for s in ids:
        if s == blank_idx: continue
        if not out or s != out[-1]: out.append(s)
    return out

def preprocess_via_dataset(pil_img: Image.Image) -> torch.Tensor:
    tmp_dir = tempfile.mkdtemp(prefix="ocr_app_")
    try:
        images_dir = os.path.join(tmp_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        img_path = os.path.join(images_dir, "tmp.png")
        pil_img.save(img_path)

        with open(os.path.join(tmp_dir, "transcripts.txt"), "w", encoding="utf-8") as f:
            f.write("images/tmp.png\t\n")  # no target text, just the path

        ds = LineDataset(tmp_dir)
        x, _ = ds[0]
        return x
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def ocr_predict(img: Image.Image):
    if img is None:
        return ""
    x = preprocess_via_dataset(img).unsqueeze(0).to(device)
    with torch.no_grad():
        lp = model(x)
        ids_t = lp[0].argmax(dim=-1).cpu().tolist()
        ids = ctc_collapse(ids_t, tok.blank_index)
        return tok.ids_to_text(ids)

demo = gr.Interface(
    fn=ocr_predict,
    inputs=gr.Image(type="pil", label="Upload a Rigveda line (clean)"),
    outputs=gr.Textbox(label="Prediction"),
    title="Rigveda OCR — CRNN (Stage-1 Clean) • exact dataset preprocessing",
)

if __name__ == "__main__":
    demo.launch()

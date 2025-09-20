# Rigveda OCR — CRNN (Stage-1 Clean)

This project demonstrates a **Stage-1 OCR pipeline** for Rigveda shlokas using a **CRNN model** trained on synthetic clean data.  
It also includes a **Fidelity Checker** to evaluate whether OCR predictions preserve the original meaning against reference translations.

---

## ✨ Features

- **OCR**: CRNN model trained on synthetic Rigveda line images (Stage-1 clean).
- **Metrics**: Character Error Rate (CER) and Word Error Rate (WER) tracking during training and validation.
- **Fidelity Checker v0**: Offline semantic + key-term similarity (option to extend with LLMs).
- **Gradio App**: Upload a Rigveda line image and see OCR predictions instantly.

---

## 📂 Layout
```
rigveda-ocr/
│
├── data/
│ └── S1_clean/ # Stage-1 clean synthetic dataset
│ ├── train/
│ │ ├── images/ # synthetic line images
│ │ └── transcripts.txt # ground truth
│ └── val/
│ ├── images/
│ └── transcripts.txt
│
├── labels/
│ └── classes_vedic.json # frozen token list (must include <pad_blank>)
│
├── runs/
│ └── crnn/ # training checkpoints + logs
│
├── src/
│ ├── crnn_model.py # CRNN model definition
│ ├── train_crnn.py # training loop (with CER/WER logging)
│ ├── eval_cer_wer_crnn.py # compute CER/WER on splits
│ ├── tokenizer.py # token encoding/decoding
│ ├── preproc.py # resize/pad + normalization
│ ├── app_crnn.py # Gradio app for live OCR
│ ├── fidelity_checker.py # offline fidelity score
│ └── ...
│
├── tools/
│ ├── inspect_val_preds.py # print GT vs PD for validation samples
│ └── predict_one.py # quick test on one image
│
├── requirements.txt
├── .gitignore
└── README.md
```
## ⚡ Setup

```bash
# clone repo
git clone https://github.com/SuperNVicious/rigveda-ocr.git
cd rigveda-ocr

# create and activate venv
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

# install dependencies
pip install -r requirements.txt
🚀 Training
bash
Copy code
python -m src.train_crnn \
  --data_root data\S1_clean \
  --labels labels\classes_vedic.json \
  --out runs\crnn\rigveda_s1_crnn.pt \
  --epochs 60 \
  --batch_size 8 \
  --lr 5e-4
Training logs show loss, Train CER/WER, and Val CER/WER.
Best model is auto-saved as rigveda_s1_crnn_best.pt.

📊 Evaluation
bash
Copy code
# Evaluate on validation split
python -m src.eval_cer_wer_crnn \
  --ckpt runs\crnn\rigveda_s1_crnn_best.pt \
  --split val
Inspect predictions side-by-side:

bash
Copy code
python -m tools.inspect_val_preds
🖼 Gradio Demo
Launch interactive OCR app:

bash
Copy code
python src\app_crnn.py
Upload a Rigveda line image → see live OCR prediction.

✅ Fidelity Checker
Run semantic + key-term fidelity scoring:

bash
Copy code
python -m src.fidelity_checker \
  --pred runs\ocr_output.txt \
  --ref  data\S1_clean\val\transcripts.txt
(LLM-based checker can be added later for transcreation validation.)

🔍 Known Issues
Predictions are excellent on Stage-1 clean data.
Generalization to noisy / scanned Rigveda pages requires Stage-2 (distortions), Stage-3 (stone effects), Stage-4 (mixed).

Current fidelity checker is offline only; LLM integration optional.

App may mispredict if uploaded images use fonts never seen in training.

📜 License
MIT License (2025) — Niranjan Chennakrishnasrinivasan
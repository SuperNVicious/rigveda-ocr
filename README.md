
# Rigveda S1 Demo (OCR + Fidelity Checker)

This scaffold shows an end-to-end **S1-only** pipeline:
- OCR (ViT-CTC greedy placeholder) — replace with your real checkpoint
- Fidelity checker v0 (offline): semantic-ish + key-term-aware score
- Gradio app for one-click demo

## Layout
```
rigveda_s1_demo/
├─ data/S1_clean/{train,val}/images/         # your S1 images
│  └─ transcripts.txt                           # TSV: relative_path\tground_truth (optional for eval)
├─ labels/
│  ├─ classes_vedic.json                        # FLAT, frozen list of tokens (replace with YOURS)
│  └─ rigveda_keylex.json                       # Sanskrit↔English key terms
├─ runs/vit_ctc/rigveda_s1.pt                   # put your real checkpoint here
├─ lm/                                          # optional later
├─ src/
│  ├─ preproc.py                                # letterbox H=48, W=512 (match your training)
│  ├─ ocr_vit_ctc.py                            # inference wrapper (stub with dummy if model missing)
│  ├─ decode_greedy.py                          # CLI greedy decode over a folder
│  ├─ eval_cer_wer.py                           # prints CER if refs present
│  ├─ fidelity_checker.py                       
│  └─ app.py                                    # Gradio demo
└─ out/
```

## Quick Start
1. **Replace classes**: edit `labels/classes_vedic.json` with your frozen FLAT list used in S1 training.
2. **Add checkpoint**: copy your S1 model to `runs/vit_ctc/rigveda_s1.pt`.
3. **Put images**: add a few S1 `val/images/*.png` and `val/transcripts.txt` for testing.

### Run OCR (greedy) over val set
```
python -m rigveda_s1_demo.src.decode_greedy --images_root rigveda_s1_demo/data/S1_clean/val/images   --transcripts rigveda_s1_demo/data/S1_clean/val/transcripts.txt   --classes rigveda_s1_demo/labels/classes_vedic.json   --checkpoint rigveda_s1_demo/runs/vit_ctc/rigveda_s1.pt   --out rigveda_s1_demo/out/s1_greedy.tsv
```

### Evaluate CER
```
python -m rigveda_s1_demo.src.eval_cer_wer --pred rigveda_s1_demo/out/s1_greedy.tsv
```

### Launch the demo app
```
python -m rigveda_s1_demo.src.app
```

> If you haven't placed your real model, the OCR box will show `[MODEL_MISSING]`. Once your checkpoint is present and you wire your actual ViT-CTC model forward pass, you'll get real text.

## Notes
- Keep **H=48, W=512** and **classes ordering** exactly as in training.
- `rigveda_keylex.json` can be expanded with more terms & synonyms.
- The LLM verdict is optional and not wired in this minimal offline build; you can add a toggle later.



## Optional: Enable LLM Transcreation Judge
Set your API key in the environment and install the SDK:
```
set OPENAI_API_KEY=sk-...          # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
pip install openai
```
Then in the app, tick **“Use LLM transcreation judge (optional)”**.  
The LLM returns **High / Medium / Low** with a 1–2 line rationale, and the UI shows a **Hybrid Score** = 0.8×offline + 0.2×LLM.

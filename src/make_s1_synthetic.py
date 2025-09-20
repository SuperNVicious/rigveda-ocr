# S1 Synthetic Line Generator (fonts -> images + transcripts.txt)
# Examples (PowerShell):
#   python src\make_s1_synthetic.py --corpus data\S1_clean\train_corpus.txt --out_dir data\S1_clean\train --fonts_dir data\fonts --render_all_fonts --repeat 2
#   python src\make_s1_synthetic.py --corpus data\S1_clean\val_corpus.txt   --out_dir data\S1_clean\val   --fonts_dir data\fonts --render_all_fonts
#
import argparse, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def list_fonts(fonts_dir: Path):
    fonts = []
    for ext in ("*.ttf","*.otf","*.ttc"):
        fonts += list(fonts_dir.glob(ext))
    if not fonts:
        raise FileNotFoundError(f"No fonts found in {fonts_dir}. Put Devanagari fonts there.")
    return sorted(fonts)

def render_line(text: str, font_path: Path, base_font_size: int = 64, pad: int = 16) -> Image.Image:
    font = ImageFont.truetype(str(font_path), base_font_size)
    dummy = Image.new("L", (1,1), 255)
    d = ImageDraw.Draw(dummy)
    _, _, w, h = d.textbbox((0,0), text, font=font)
    W = max(256, w + 2*pad)
    H = max(96,  h + 2*pad)
    img = Image.new("L", (W,H), 255)
    d = ImageDraw.Draw(img)
    d.text((pad, pad), text, font=font, fill=0)
    return img

def letterbox(im: Image.Image, out_h: int = 48, out_w: int = 512) -> Image.Image:
    w, h = im.size
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    im_resized = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (out_w, out_h), 255)
    left = (out_w - nw)//2
    top  = (out_h - nh)//2
    canvas.paste(im_resized, (left, top))
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fonts_dir", required=True)
    ap.add_argument("--font_size", type=int, default=64)
    ap.add_argument("--render_all_fonts", action="store_true", help="render each line in all fonts (multiplies size)")
    ap.add_argument("--repeat", type=int, default=1, help="repeat each (line x font) this many times (>=1)")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    out_dir = Path(args.out_dir)
    fonts_dir = Path(args.fonts_dir)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # BOM-safe read; strip BOM if present
    raw_lines = corpus_path.read_text(encoding="utf-8-sig").splitlines()
    lines = [ln.strip().lstrip("\ufeff") for ln in raw_lines if ln.strip()]
    if not lines:
        print(f"No lines in {corpus_path}.")
        (out_dir / "transcripts.txt").write_text("", encoding="utf-8")
        return

    fonts = list_fonts(fonts_dir)

    idx = 1
    records = []
    for li, text in enumerate(lines):
        font_list = fonts if args.render_all_fonts else [random.choice(fonts)]
        for fi, font_path in enumerate(font_list):
            for rep in range(max(1, args.repeat)):
                big = render_line(text, font_path, base_font_size=args.font_size)
                final = letterbox(big, 48, 512)
                fname = f"{idx:04d}_f{fi:02d}_r{rep:02d}.png" if args.render_all_fonts or args.repeat>1 else f"{idx:04d}.png"
                final.save(out_images / fname)
                records.append((f"images/{fname}", text))
                idx += 1

    with open(out_dir / "transcripts.txt", "w", encoding="utf-8") as f:
        for p, t in records:
            f.write(f"{p}\t{t}\n")
    print(f"Generated {len(records)} images at {out_images} and transcripts.txt")

if __name__ == "__main__":
    main()

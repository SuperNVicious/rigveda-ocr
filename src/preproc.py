
import cv2
import numpy as np

H, W = 48, 512  # keep consistent with training
def letterbox(img, h=H, w=W):
    # convert to grayscale, resize with aspect, then pad
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H0, W0 = img.shape[:2]
    scale = min(w / W0, h / H0)
    newW, newH = int(W0 * scale), int(H0 * scale)
    img_resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w), 255, dtype=img.dtype)
    y0 = (h - newH) // 2
    x0 = (w - newW) // 2
    canvas[y0:y0+newH, x0:x0+newW] = img_resized
    img_norm = canvas.astype(np.float32) / 255.0
    img_norm = (img_norm - 0.5) / 0.5  # simple normalization
    return img_norm

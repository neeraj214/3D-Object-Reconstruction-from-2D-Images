import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def resize_aspect(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (3, 3), 0)


def threshold(img: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_image(path: str) -> np.ndarray:
    img = load_image(path)
    g = to_grayscale(img)
    r = resize_aspect(g)
    d = denoise(r)
    t = threshold(d)
    return t

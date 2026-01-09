import argparse
from pathlib import Path
import numpy as np
import cv2

def resize_normalize(img, size=(384,384)):
    x = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32) / 255.0
    return x

def grabcut_remove_bg(img):
    h, w = img.shape[:2]
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        res = img * m[:, :, None]
        return res
    except Exception:
        return img

def sobel_edge(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx*gx + gy*gy)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    img = cv2.imread(args.image)
    img2 = grabcut_remove_bg(img)
    img3 = (resize_normalize(img2) * 255).astype(np.uint8)
    edge = sobel_edge(img)
    cv2.imwrite(args.out, img3)
    cv2.imwrite(str(Path(args.out).with_name('edge.png')), (edge*255).astype(np.uint8))

if __name__ == '__main__':
    main()
import argparse
from pathlib import Path
import numpy as np
import csv
from src.utils.metrics import chamfer_distance

def evaluate(pred_dir, gt_dir, out_csv):
    Path(Path(out_csv).parent).mkdir(parents=True, exist_ok=True)
    rows = []
    for pred_file in Path(pred_dir).glob('*.npy'):
        gt_file = Path(gt_dir) / pred_file.name
        if gt_file.exists():
            a = np.load(pred_file).astype(np.float32)
            b = np.load(gt_file).astype(np.float32)
            cd = chamfer_distance(a, b)
            rows.append([pred_file.name, cd])
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file','chamfer'])
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=str, required=True)
    ap.add_argument('--gt', type=str, required=True)
    ap.add_argument('--out', type=str, default='results/metrics.csv')
    args = ap.parse_args()
    evaluate(args.pred, args.gt, args.out)

if __name__ == '__main__':
    main()
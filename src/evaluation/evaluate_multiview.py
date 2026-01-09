import argparse
import json
from pathlib import Path
import numpy as np

def summarize(metrics_path):
    d = json.loads(Path(metrics_path).read_text())
    items = d.get("items", [])
    cd = [x.get("cd", 0.0) for x in items]
    fs = [x.get("fscore", 0.0) for x in items]
    iou = [x.get("iou", 0.0) for x in items]
    cats = {}
    for x in items:
        c = x.get("category", "unknown")
        cats.setdefault(c, {"cd": [], "fscore": [], "iou": []})
        cats[c]["cd"].append(x.get("cd", 0.0))
        cats[c]["fscore"].append(x.get("fscore", 0.0))
        cats[c]["iou"].append(x.get("iou", 0.0))
    cat_summary = {c: {
        "avg_cd": float(np.mean(v["cd"])) if v["cd"] else 0.0,
        "avg_fscore": float(np.mean(v["fscore"])) if v["fscore"] else 0.0,
        "avg_iou": float(np.mean(v["iou"])) if v["iou"] else 0.0
    } for c, v in cats.items()}
    return {
        "avg_cd": float(np.mean(cd)) if cd else 0.0,
        "avg_fscore": float(np.mean(fs)) if fs else 0.0,
        "avg_iou": float(np.mean(iou)) if iou else 0.0,
        "categorywise": cat_summary
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True)
    args = ap.parse_args()
    r = summarize(args.metrics)
    print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
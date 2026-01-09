import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

try:
    from src.datasets.unified_multiview import UnifiedMultiviewDataset
    from src.models.multiview_pointcloud import MultiViewPointCloudModel
    from src.training.losses import silhouette_iou_loss
except Exception:
    UnifiedMultiviewDataset = None
    MultiViewPointCloudModel = None
    silhouette_iou_loss = None


def plot_completeness_and_heatmap(batch_json: Path, out_dir: Path):
    data = json.loads(batch_json.read_text())
    items = data.get("items", [])
    if not items:
        return
    comp = [it.get("completeness") for it in items if it.get("completeness") is not None]
    p2s = [it.get("point_to_surface") for it in items if it.get("point_to_surface") is not None]
    nc = [it.get("normal_consistency") for it in items if it.get("normal_consistency") is not None]
    out_dir.mkdir(parents=True, exist_ok=True)
    if comp:
        plt.figure(); plt.hist(comp, bins=20, color="seagreen"); plt.title("Completeness"); plt.tight_layout(); plt.savefig(out_dir/"completeness_hist.png"); plt.close()
    if p2s and nc:
        H, xedges, yedges = np.histogram2d(p2s, nc, bins=20)
        plt.figure(); plt.imshow(H.T, origin="lower", aspect="auto", cmap="magma"); plt.xlabel("point_to_surface"); plt.ylabel("normal_consistency"); plt.colorbar(); plt.tight_layout(); plt.savefig(out_dir/"mesh_quality_heatmap.png"); plt.close()


def plot_silhouette_iou_hist(ckpt: Path, ann: Path, out_dir: Path, max_batches: int = 8):
    if UnifiedMultiviewDataset is None:
        return None
    ds = UnifiedMultiviewDataset(str(ann), split="val", num_points=2048, image_size=224, augment=False)
    if len(ds) == 0:
        return None
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    state = torch.load(str(ckpt), map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model = MultiViewPointCloudModel(num_points=2048)
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device); model.eval()
    vals = []
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            f = batch["front"].to(device)
            s = batch.get("side", f).to(device)
            pts = model(f, s)
            if "mask" in batch and "camera_K" in batch:
                try:
                    si = silhouette_iou_loss(pts, batch["camera_K"].to(device), batch["mask"].to(device))
                    si = (1.0 - si).detach().cpu().numpy()
                    vals.extend(list(si))
                except Exception:
                    pass
            if bi >= max_batches - 1:
                break
    if not vals:
        return None
    arr = np.array(vals)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.hist(arr, bins=20, color="dodgerblue"); plt.title("Silhouette IoU (val)"); plt.tight_layout(); plt.savefig(out_dir/"silhouette_iou_hist.png"); plt.close()
    return float(arr.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", type=str, required=True)
    ap.add_argument("--batch-json", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=False)
    ap.add_argument("--ann", type=str, default="data/unified/annotations.json")
    args = ap.parse_args()
    out_dir = Path(args.round_dir)
    plot_completeness_and_heatmap(Path(args.batch_json), out_dir)
    sil_mean = None
    if args.ckpt:
        sil_mean = plot_silhouette_iou_hist(Path(args.ckpt), Path(args.ann), out_dir)
    summ_p = out_dir/"summary.json"
    base = {}
    if summ_p.exists():
        try:
            base = json.loads(summ_p.read_text())
        except Exception:
            base = {}
    if sil_mean is not None:
        base["silhouette_iou_mean"] = sil_mean
        summ_p.write_text(json.dumps(base, indent=2))


if __name__ == "__main__":
    main()
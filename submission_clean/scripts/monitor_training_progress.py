import time
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.unified_multiview import UnifiedMultiviewDataset
from src.models.multiview_pointcloud import MultiViewPointCloudModel
from src.training.losses import silhouette_iou_loss, point_to_surface_loss, normal_consistency_loss


def completeness_pc(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 0.02) -> torch.Tensor:
    d = torch.cdist(gt, pred, p=2).min(dim=1)[0]
    return (d < thresh).float().mean()


def compute_progress_metrics(annotations_path: str, ckpt_path: Path, out_dir: Path, num_points: int = 2048, max_batches: int = 8) -> Dict[str, Any]:
    ds = UnifiedMultiviewDataset(annotations_path, split="val", num_points=num_points, image_size=224, augment=False)
    if len(ds) == 0:
        return {"count": 0}
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    model = MultiViewPointCloudModel(num_points=num_points)
    state = torch.load(str(ckpt_path), map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    items = []
    with torch.no_grad():
        count = 0
        for batch in dl:
            f = batch["front"].to(device)
            s = batch.get("side", f).to(device)
            gt = batch["pointcloud"].to(device).permute(0, 2, 1)
            pts = model(f, s)
            sil = None
            if "mask" in batch and "camera_K" in batch:
                try:
                    sil = silhouette_iou_loss(pts, batch["camera_K"].to(device), batch["mask"].to(device)).detach().cpu()
                except Exception:
                    sil = None
            p2s_vals = []
            if "mesh_vertices" in batch:
                for i, mv in enumerate(batch["mesh_vertices"]):
                    if mv is None:
                        continue
                    mv_t = mv.to(device).unsqueeze(0)
                    pv_t = pts[i].unsqueeze(0)
                    try:
                        p2s_vals.append(point_to_surface_loss(pv_t, mv_t).mean().detach().cpu())
                    except Exception:
                        pass
            ncons = normal_consistency_loss(pts, gt, k=16).detach().cpu()
            comp_vals = []
            for i in range(pts.size(0)):
                comp_vals.append(completeness_pc(pts[i], gt[i]).detach().cpu())
            for i in range(pts.size(0)):
                items.append({
                    "silhouette_iou": float((1.0 - sil[i].item()) if sil is not None else 0.0),
                    "point_to_surface": float(p2s_vals[i].item()) if i < len(p2s_vals) else None,
                    "normal_consistency": float(ncons[i].item()),
                    "completeness": float(comp_vals[i].item()),
                })
            count += 1
            if count >= max_batches:
                break
    agg = {}
    if items:
        def filt(key):
            vals = [it[key] for it in items if it[key] is not None]
            return float(np.mean(vals)) if vals else None
        agg = {
            "count": len(items),
            "silhouette_iou_mean": filt("silhouette_iou"),
            "point_to_surface_mean": filt("point_to_surface"),
            "normal_consistency_mean": filt("normal_consistency"),
            "completeness_mean": filt("completeness"),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"aggregate": agg, "items": items}


def write_curve(log_path: Path, out_png: Path):
    try:
        import matplotlib.pyplot as plt
        if log_path.exists():
            data = json.loads(log_path.read_text())
            xs = [d.get("epoch", i + 1) for i, d in enumerate(data)]
            ys = [d.get("loss", 0.0) for d in data]
            plt.figure()
            plt.plot(xs, ys, "-o")
            plt.title("Stage2 Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_png)
            plt.close()
    except Exception:
        pass


def main():
    out_root = Path("results/checkpoints_multiview")
    stage2 = out_root / "stage2"
    progress_dir = Path("results/metrics_progress")
    progress_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    while True:
        ckpts = sorted(list(stage2.glob("stage2_epoch*.pth")))
        for ck in ckpts:
            name = ck.stem
            try:
                ep = int(name.split("epoch")[-1])
            except Exception:
                continue
            if ep % 5 == 0 and name not in seen:
                seen.add(name)
                metrics = compute_progress_metrics("data/unified/annotations.json", ck, progress_dir)
                out_json = progress_dir / f"ep_{ep}.json"
                out_json.write_text(json.dumps(metrics, indent=2))
                write_curve(stage2 / "training_log.json", progress_dir / f"loss_curve_ep_{ep}.png")
                agg = metrics.get("aggregate", {})
                plan = {"epoch": ep, "suggested": {}}
                # Simple heuristics to improve accuracy
                lw = {
                    "chamfer": 0.5,
                    "fscore": 0.2,
                    "normal": 0.15,
                    "laplacian": 0.1,
                    "perceptual": 0.05,
                    "p2s": 0.15,
                    "silhouette": 0.1,
                }
                si = agg.get("silhouette_iou_mean") or 0.0
                p2s = agg.get("point_to_surface_mean") or 0.0
                nc = agg.get("normal_consistency_mean") or 0.0
                if si < 0.2:
                    lw["silhouette"] = 0.2
                if p2s > 0.05:
                    lw["p2s"] = 0.25
                if nc < 0.5:
                    lw["normal"] = 0.2
                plan["suggested"]["loss_weights"] = lw
                plan["suggested"]["epochs2"] = 20 if ep >= 10 else 15
                (progress_dir / f"adjustment_plan_ep_{ep}.json").write_text(json.dumps(plan, indent=2))
                # Write adjusted config for potential next run
                adj_cfg = {
                    "epochs": 35,
                    "batch": 8,
                    "points_coarse": 1024,
                    "points_refined": 4096,
                    "lr": 0.0005,
                    "lr_schedule": "cosine",
                    "loss_weights": lw,
                }
                Path("configs/train_config_adjusted.json").write_text(json.dumps(adj_cfg, indent=2))
        time.sleep(30)


if __name__ == "__main__":
    main()
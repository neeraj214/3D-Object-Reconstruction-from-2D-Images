import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from src.datasets.unified_multiview import UnifiedMultiviewDataset
from src.models.multiview_pointcloud import MultiViewPointCloudModel
from src.training.losses import silhouette_iou_loss

BASE = Path("results/continuous_training")
BASE.mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def latest_stage2_ckpt(root: Path) -> Optional[Path]:
    s2 = root / "stage2"
    cks = sorted(s2.glob("stage2_epoch*.pth"))
    if cks:
        return cks[-1]
    bp = s2 / "best.pth"
    return bp if bp.exists() else None


def read_epochs(stage_dir: Path) -> int:
    log = stage_dir / "training_log.json"
    if not log.exists():
        return 0
    try:
        data = json.loads(log.read_text())
        return int(max([d.get("epoch", 0) for d in data] or [0]))
    except Exception:
        return 0


def run_eval_batch(ckpt: Path, limit: int, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "mesh_eval_batch.json"
    cmd = [
        "python", "scripts/eval_mesh_batch.py",
        "--limit", str(limit),
        "--out", str(out_json),
        "--use-model",
        "--ckpt", str(ckpt),
    ]
    subprocess.run(cmd, check=False)
    data = read_json(out_json) or {"items": [], "aggregate": {}}
    return data


def plot_distributions(data: Dict[str, Any], out_dir: Path):
    items = data.get("items", [])
    if not items:
        return
    # Silhouette IoU histogram
    si = [it.get("silhouette_iou") for it in items if it.get("silhouette_iou") is not None]
    plt.figure(); plt.hist(si, bins=20, color="steelblue"); plt.title("Silhouette IoU"); plt.tight_layout(); plt.savefig(out_dir / "silhouette_iou_hist.png"); plt.close()
    # Completeness histogram
    comp = [it.get("completeness") for it in items if it.get("completeness") is not None]
    plt.figure(); plt.hist(comp, bins=20, color="seagreen"); plt.title("Completeness"); plt.tight_layout(); plt.savefig(out_dir / "completeness_hist.png"); plt.close()
    # Mesh quality heatmap (p2s vs normal consistency)
    p2s = [it.get("point_to_surface") for it in items if it.get("point_to_surface") is not None]
    nc = [it.get("normal_consistency") for it in items if it.get("normal_consistency") is not None]
    if p2s and nc:
        H, xedges, yedges = np.histogram2d(p2s, nc, bins=20)
        plt.figure(); plt.imshow(H.T, origin="lower", aspect="auto", cmap="magma"); plt.xlabel("point_to_surface"); plt.ylabel("normal_consistency"); plt.colorbar(); plt.tight_layout(); plt.savefig(out_dir / "mesh_quality_heatmap.png"); plt.close()


def eval_silhouette_distribution(ckpt: Path, limit_batches: int = 8) -> Optional[np.ndarray]:
    ds = UnifiedMultiviewDataset("data/unified/annotations.json", split="val", num_points=2048, image_size=224, augment=False)
    if len(ds) == 0:
        return None
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    state = torch.load(str(ckpt), map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model = MultiViewPointCloudModel(num_points=2048)
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    vals = []
    with torch.no_grad():
        c = 0
        for batch in dl:
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
            c += 1
            if c >= limit_batches:
                break
    if len(vals) == 0:
        return None
    return np.array(vals)


def suggest_adjustments(agg: Dict[str, Any]) -> Dict[str, Any]:
    si = agg.get("silhouette_iou_mean") or 0.0
    p2s = agg.get("point_to_surface_mean") or 0.0
    nc = agg.get("normal_consistency_mean") or 0.0
    lw = {
        "chamfer": 0.5,
        "fscore": 0.2,
        "normal": 0.15,
        "laplacian": 0.1,
        "perceptual": 0.05,
        "p2s": 0.15,
        "silhouette": 0.1,
    }
    if si < 0.3:
        lw["silhouette"] = 0.2
    if p2s > 0.05:
        lw["p2s"] = 0.25
    if nc < 0.6:
        lw["normal"] = 0.2
    return {
        "epochs1": 15,
        "epochs2": 20,
        "lr": 8e-5,
        "loss_weights": lw,
        "lr_schedule": "cosine",
    }


def write_adjusted_config(cfg: Dict[str, Any], out_path: Path):
    payload = {
        "epochs": cfg.get("epochs1", 15) + cfg.get("epochs2", 20),
        "batch": 8,
        "points_coarse": 1024,
        "points_refined": 4096,
        "lr": cfg.get("lr", 5e-4),
        "lr_schedule": cfg.get("lr_schedule", "cosine"),
        "loss_weights": cfg.get("loss_weights", {}),
    }
    out_path.write_text(json.dumps(payload, indent=2))


def training_idle(stage_dir: Path, expected_epochs: int) -> bool:
    # Consider idle if we have reached expected epochs and no newer file in 5 minutes
    ep = read_epochs(stage_dir)
    if ep < expected_epochs:
        return False
    latest = max([p.stat().st_mtime for p in stage_dir.glob("*.pth")] or [0])
    return (time.time() - latest) > 300


def start_retrain(adjust_cfg: Path):
    cmd = [
        "python", "src/training/train_multiview.py",
        "--annotations", "data/unified/annotations.json",
        "--out", "results/checkpoints_multiview",
        "--points", "2048",
        "--epochs1", "15",
        "--epochs2", "20",
        "--batch", "8",
        "--lr", "1e-4",
        "--config", str(adjust_cfg),
    ]
    # Fire-and-forget; external orchestrator manages terminals
    subprocess.Popen(cmd)


def main():
    root = Path("results/checkpoints_multiview")
    stage2 = root / "stage2"
    rounds = 0
    target_accuracy = 0.75
    expected_epochs2 = 15
    state_p = BASE / "state.json"
    while True:
        ckpt = latest_stage2_ckpt(root)
        if ckpt is None:
            time.sleep(60)
            continue
        rounds += 1
        rdir = BASE / f"round_{rounds}"
        rdir.mkdir(parents=True, exist_ok=True)
        data = run_eval_batch(ckpt, limit=12, out_dir=rdir)
        plot_distributions(data, rdir)
        # Silhouette IoU histogram from validation set with current checkpoint
        sil_vals = eval_silhouette_distribution(ckpt)
        if sil_vals is not None:
            plt.figure(); plt.hist(sil_vals, bins=20, color="dodgerblue"); plt.title("Silhouette IoU (val)"); plt.tight_layout(); plt.savefig(rdir / "silhouette_iou_hist.png"); plt.close()
            summary_sil = float(np.mean(sil_vals))
        else:
            summary_sil = None
        agg = data.get("aggregate", {})
        summary = {
            "round": rounds,
            "ckpt": str(ckpt),
            "aggregate": agg,
            "silhouette_iou_mean": summary_sil,
        }
        (rdir / "summary.json").write_text(json.dumps(summary, indent=2))
        prev = read_json(state_p) or {}
        prev_score = (prev.get("aggregate", {}) or {}).get("normal_consistency_mean") or 0.0
        curr_score = agg.get("normal_consistency_mean") or 0.0
        stable = abs(curr_score - prev_score) < 1e-3
        achieved = curr_score >= target_accuracy
        state_p.write_text(json.dumps(summary, indent=2))
        if achieved or stable:
            # Stop if stable or target reached
            time.sleep(300)
            continue
        # Prepare adjustments
        cfg = suggest_adjustments(agg)
        adj_cfg = Path("configs/train_config_adjusted.json")
        write_adjusted_config(cfg, adj_cfg)
        # Retrain only if current training is idle
        if training_idle(stage2, expected_epochs2):
            start_retrain(adj_cfg)
            expected_epochs2 = cfg.get("epochs2", expected_epochs2)
        # Short sleep before next round
        time.sleep(300)


if __name__ == "__main__":
    main()
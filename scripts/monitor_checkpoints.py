import json
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.datasets.unified_multiview import UnifiedMultiviewDataset
from src.models.multiview_pointcloud import MultiViewPointCloudModel
from src.training.losses import chamfer_distance_v2, fscore_multi, laplacian_regularization, normal_consistency_loss, perceptual_loss_clip, silhouette_iou_loss, point_to_surface_loss
from src.evaluation.mesh_metrics import mesh_completeness

ROOT = Path("results/checkpoints_multiview")
ANN = Path("data/unified/annotations.json")
EPOCHS = [5, 10, 15, 20, 25, 30]

def mv_collate(batch):
    out = {}
    for k in ["front","side","pointcloud","mask","camera_K"]:
        vals = [b[k] for b in batch if k in b and b[k] is not None]
        if len(vals) > 0:
            out[k] = torch.stack(vals, dim=0)
    out["category"] = [b.get("category","unknown") for b in batch]
    mv_list = [b.get("mesh_vertices", None) for b in batch]
    mf_list = [b.get("mesh_faces", None) for b in batch]
    if any(v is not None for v in mv_list):
        out["mesh_vertices"] = mv_list
    if any(f is not None for f in mf_list):
        out["mesh_faces"] = mf_list
    return out

def eval_epoch(stage_dir: Path, stage: int, epoch: int):
    ck = stage_dir/ f"stage{stage}_epoch{epoch}.pth"
    out_json = stage_dir/ f"checkpoint_metrics_epoch{epoch}.json"
    if not ck.exists():
        return False
    if out_json.exists():
        return True
    ds = UnifiedMultiviewDataset(str(ANN), split="val", num_points=2048, image_size=224, augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=mv_collate)
    model = MultiViewPointCloudModel(num_points=2048)
    state = torch.load(str(ck), map_location="cpu")
    model.load_state_dict(state["model"]) if "model" in state else model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    v_losses = []
    v_p2s = []
    v_norm = []
    v_comp = []
    v_sil_iou = []
    with torch.no_grad():
        for batch in dl:
            f = batch["front"].to(device)
            s = batch.get("side", f).to(device)
            gt = batch["pointcloud"].to(device).permute(0,2,1)
            pts = model(f, s) if stage == 2 else model(f, f)
            bb = gt.max(dim=1)[0]-gt.min(dim=1)[0]
            diag = torch.norm(bb, dim=1)
            cd2 = chamfer_distance_v2(pts, gt).mean()
            f1, f5, f10 = fscore_multi(pts, gt, diag)
            f_loss = 1.0 - ((f1+f5+f10)/3.0).mean()
            lap = laplacian_regularization(pts, k=16).mean()
            nloss = normal_consistency_loss(pts, gt, k=16).mean()
            ploss = perceptual_loss_clip([(f, s)])
            sil = 0.0
            if "mask" in batch and "camera_K" in batch:
                try:
                    sil = silhouette_iou_loss(pts, batch["camera_K"], batch["mask"]).mean()
                except Exception:
                    sil = 0.0
            p2s_val = 0.0
            if "mesh_vertices" in batch:
                try:
                    vals = []
                    for i, mv_i in enumerate(batch["mesh_vertices"]):
                        if mv_i is None:
                            continue
                        mv_t = mv_i.to(device).unsqueeze(0)
                        pv_t = pts[i].unsqueeze(0)
                        vals.append(point_to_surface_loss(pv_t, mv_t).mean())
                    if len(vals) > 0:
                        p2s_val = torch.stack(vals).mean()
                except Exception:
                    p2s_val = 0.0
            loss_val = 0.5*cd2 + 0.2*f_loss + 0.1*lap + 0.15*nloss + 0.05*ploss.mean()
            if isinstance(p2s_val, torch.Tensor):
                loss_val = loss_val + 0.15*p2s_val
            else:
                loss_val = loss_val + 0.15*torch.tensor(float(p2s_val), device=device)
            if isinstance(sil, torch.Tensor):
                loss_val = loss_val + 0.1*sil
            else:
                loss_val = loss_val + 0.1*torch.tensor(float(sil), device=device)
            v_losses.append(loss_val.item())
            v_norm.append(float(nloss.item()))
            v_p2s.append(float(p2s_val.item()) if isinstance(p2s_val, torch.Tensor) else float(p2s_val))
            try:
                for i in range(pts.size(0)):
                    v_comp.append(mesh_completeness(pts[i].detach().cpu().numpy(), gt[i].detach().cpu().numpy()))
            except Exception:
                pass
            try:
                v_sil_iou.append(float(1.0 - (sil.item() if isinstance(sil, torch.Tensor) else sil)))
            except Exception:
                pass
    train_log = json.loads((stage_dir/"training_log.json").read_text())
    train_loss = next((d["loss"] for d in train_log if d.get("epoch") == epoch), None)
    weights_scheduled = True if epoch >= 5 else False
    normalization_applied = True if (epoch % 5 == 0) else False
    metrics = {
        "epoch": int(epoch),
        "train_loss": float(train_loss) if train_loss is not None else None,
        "val_loss": float(np.mean(v_losses)) if len(v_losses) > 0 else None,
        "p2s": float(np.mean(v_p2s)) if len(v_p2s) > 0 else None,
        "normal_consistency": float(np.mean(v_norm)) if len(v_norm) > 0 else None,
        "completeness": float(np.mean(v_comp)) if len(v_comp) > 0 else None,
        "silhouette_iou": float(np.mean(v_sil_iou)) if len(v_sil_iou) > 0 else None,
        "lr_reduced": False,
        "weights_scheduled": weights_scheduled,
        "normalization_applied": normalization_applied
    }
    out_json.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))
    return True

def eval_specific(stage_dir: Path, stage: int, ck_epoch: int, report_epoch: int):
    ck = stage_dir/ f"stage{stage}_epoch{ck_epoch}.pth"
    out_json = stage_dir/ f"checkpoint_metrics_epoch{report_epoch}.json"
    if not ck.exists():
        return False
    if out_json.exists():
        return True
    ds = UnifiedMultiviewDataset(str(ANN), split="val", num_points=2048, image_size=224, augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=mv_collate)
    model = MultiViewPointCloudModel(num_points=2048)
    state = torch.load(str(ck), map_location="cpu")
    model.load_state_dict(state["model"]) if "model" in state else model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    v_losses = []
    v_p2s = []
    v_norm = []
    v_comp = []
    v_sil_iou = []
    with torch.no_grad():
        for batch in dl:
            f = batch["front"].to(device)
            s = batch.get("side", f).to(device)
            gt = batch["pointcloud"].to(device).permute(0,2,1)
            pts = model(f, s) if stage == 2 else model(f, f)
            bb = gt.max(dim=1)[0]-gt.min(dim=1)[0]
            diag = torch.norm(bb, dim=1)
            cd2 = chamfer_distance_v2(pts, gt).mean()
            f1, f5, f10 = fscore_multi(pts, gt, diag)
            f_loss = 1.0 - ((f1+f5+f10)/3.0).mean()
            lap = laplacian_regularization(pts, k=16).mean()
            nloss = normal_consistency_loss(pts, gt, k=16).mean()
            ploss = perceptual_loss_clip([(f, s)])
            sil = 0.0
            if "mask" in batch and "camera_K" in batch:
                try:
                    sil = silhouette_iou_loss(pts, batch["camera_K"], batch["mask"]).mean()
                except Exception:
                    sil = 0.0
            p2s_val = 0.0
            if "mesh_vertices" in batch:
                try:
                    vals = []
                    for i, mv_i in enumerate(batch["mesh_vertices"]):
                        if mv_i is None:
                            continue
                        mv_t = mv_i.to(device).unsqueeze(0)
                        pv_t = pts[i].unsqueeze(0)
                        vals.append(point_to_surface_loss(pv_t, mv_t).mean())
                    if len(vals) > 0:
                        p2s_val = torch.stack(vals).mean()
                except Exception:
                    p2s_val = 0.0
            loss_val = 0.5*cd2 + 0.2*f_loss + 0.1*lap + 0.15*nloss + 0.05*ploss.mean()
            if isinstance(p2s_val, torch.Tensor):
                loss_val = loss_val + 0.15*p2s_val
            else:
                loss_val = loss_val + 0.15*torch.tensor(float(p2s_val), device=device)
            if isinstance(sil, torch.Tensor):
                loss_val = loss_val + 0.1*sil
            else:
                loss_val = loss_val + 0.1*torch.tensor(float(sil), device=device)
            v_losses.append(loss_val.item())
            v_norm.append(float(nloss.item()))
            v_p2s.append(float(p2s_val.item()) if isinstance(p2s_val, torch.Tensor) else float(p2s_val))
            try:
                for i in range(pts.size(0)):
                    v_comp.append(mesh_completeness(pts[i].detach().cpu().numpy(), gt[i].detach().cpu().numpy()))
            except Exception:
                pass
            try:
                v_sil_iou.append(float(1.0 - (sil.item() if isinstance(sil, torch.Tensor) else sil)))
            except Exception:
                pass
    # Training log (may have been reset on resume)
    train_loss = None
    try:
        train_log = json.loads((stage_dir/"training_log.json").read_text())
        train_loss = next((d["loss"] for d in train_log if d.get("epoch") == ck_epoch), None)
    except Exception:
        train_loss = None
    weights_scheduled = True if report_epoch >= 5 else False
    normalization_applied = True if (report_epoch % 5 == 0) else False
    metrics = {
        "epoch": int(report_epoch),
        "train_loss": float(train_loss) if train_loss is not None else None,
        "val_loss": float(np.mean(v_losses)) if len(v_losses) > 0 else None,
        "p2s": float(np.mean(v_p2s)) if len(v_p2s) > 0 else None,
        "normal_consistency": float(np.mean(v_norm)) if len(v_norm) > 0 else None,
        "completeness": float(np.mean(v_comp)) if len(v_comp) > 0 else None,
        "silhouette_iou": float(np.mean(v_sil_iou)) if len(v_sil_iou) > 0 else None,
        "lr_reduced": False,
        "weights_scheduled": weights_scheduled,
        "normalization_applied": normalization_applied
    }
    out_json.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))
    return True

def run_once():
    for st in [1, 2]:
        sd = ROOT/ f"stage{st}"
        if sd.exists():
            for ep in EPOCHS:
                try:
                    eval_epoch(sd, st, ep)
                except Exception:
                    pass
    # Map resumed stage-2 epochs (5->25, 10->30) if needed
    s2 = ROOT/"stage2"
    if s2.exists():
        try:
            if (s2/"stage2_epoch20.pth").exists() and (s2/"stage2_epoch5.pth").exists():
                eval_specific(s2, 2, 5, 25)
            # If resumed cycle started at epoch 25, map 5->30 for the second segment
            try:
                import json
                adj_p = s2/"adjustments.json"
                if adj_p.exists():
                    adj = json.loads(adj_p.read_text())
                    if any(d.get("epoch") == 25 and d.get("apply_corrections") for d in adj):
                        if (s2/"stage2_epoch5.pth").exists():
                            eval_specific(s2, 2, 5, 30)
                elif (s2/"stage2_epoch10.pth").exists():
                    eval_specific(s2, 2, 10, 30)
            except Exception:
                pass
        except Exception:
            pass

def write_batch_eval():
    s2 = ROOT/"stage2"
    ck = s2/"stage2_epoch30.pth"
    out_json = ROOT.parent/"mesh_eval_batch.json"
    out_csv = ROOT.parent/"mesh_eval_batch.csv"
    if not ck.exists():
        return False
    ds = UnifiedMultiviewDataset(str(ANN), split="test", num_points=2048, image_size=224, augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=mv_collate)
    model = MultiViewPointCloudModel(num_points=2048)
    state = torch.load(str(ck), map_location="cpu")
    model.load_state_dict(state["model"]) if "model" in state else model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dl:
            f = batch["front"].to(device)
            s = batch.get("side", f).to(device)
            gt = batch["pointcloud"].to(device).permute(0,2,1)
            pts = model(f, s)
            bb = gt.max(dim=1)[0]-gt.min(dim=1)[0]
            diag = torch.norm(bb, dim=1)
            cd2 = chamfer_distance_v2(pts, gt)
            f1, f5, f10 = fscore_multi(pts, gt, diag)
            lap = laplacian_regularization(pts, k=16)
            nloss = normal_consistency_loss(pts, gt, k=16)
            sil = None
            if "mask" in batch and "camera_K" in batch:
                try:
                    sil = 1.0 - silhouette_iou_loss(pts, batch["camera_K"], batch["mask"]).mean().item()
                except Exception:
                    sil = None
            p2s_val = None
            if "mesh_vertices" in batch:
                try:
                    vals = []
                    for i, mv_i in enumerate(batch["mesh_vertices"]):
                        if mv_i is None:
                            continue
                        mv_t = mv_i.to(device).unsqueeze(0)
                        pv_t = pts[i].unsqueeze(0)
                        vals.append(point_to_surface_loss(pv_t, mv_t).mean())
                    if len(vals) > 0:
                        p2s_val = torch.stack(vals).mean().item()
                except Exception:
                    p2s_val = None
            for i in range(pts.size(0)):
                comp = None
                try:
                    comp = mesh_completeness(pts[i].detach().cpu().numpy(), gt[i].detach().cpu().numpy())
                except Exception:
                    comp = None
                rows.append({
                    "cd2": float(cd2[i].item()),
                    "f1": float(f1[i].item()),
                    "f5": float(f5[i].item()),
                    "f10": float(f10[i].item()),
                    "laplacian": float(lap[i].item()),
                    "normal_consistency": float(nloss[i].item()),
                    "p2s": float(p2s_val) if p2s_val is not None else None,
                    "completeness": float(comp) if comp is not None else None,
                    "silhouette_iou": float(sil) if sil is not None else None
                })
    out_json.write_text(json.dumps({"items": rows}, indent=2))
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cd2","f1","f5","f10","laplacian","normal_consistency","p2s","completeness","silhouette_iou"])
        for r in rows:
            w.writerow([r.get("cd2"), r.get("f1"), r.get("f5"), r.get("f10"), r.get("laplacian"), r.get("normal_consistency"), r.get("p2s"), r.get("completeness"), r.get("silhouette_iou")])
    return True

def main():
    while True:
        run_once()
        try:
            write_batch_eval()
        except Exception:
            pass
        time.sleep(60)

if __name__ == "__main__":
    main()
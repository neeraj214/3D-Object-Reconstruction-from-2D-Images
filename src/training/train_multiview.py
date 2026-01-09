import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.datasets.unified_multiview import UnifiedMultiviewDataset
from src.models.multiview_pointcloud import MultiViewPointCloudModel
from src.training.losses import chamfer_distance_v2, fscore_multi, voxel_iou, normal_consistency_loss, laplacian_regularization, perceptual_loss_clip, silhouette_iou_loss, point_to_surface_loss
from src.evaluation.mesh_metrics import mesh_completeness

def compute_intrinsics(h, w, f_scale=1.0):
    f = max(h, w) * float(f_scale)
    cx = w / 2.0
    cy = h / 2.0
    return [[f, 0, cx], [0, f, cy], [0, 0, 1.0]]

def _check_intrinsics_consistency(name, K, h, w, f_scale=1.0):
    try:
        import numpy as np
        K_ref = np.array(compute_intrinsics(h, w, f_scale=f_scale), dtype=np.float32)
        K_in = np.array(K, dtype=np.float32) if K is not None else K_ref
        match = np.allclose(K_in, K_ref, rtol=1e-5, atol=1e-5)
        print(f"[intrinsics-check] {name}: match={match}")
        if not match:
            print(f"[intrinsics-check] {name}: expected={K_ref.tolist()} got={K_in.tolist()}")
    except Exception as e:
        print(f"[intrinsics-check] {name}: failed {e}")
def train_stage(dset_path, stage, out_dir, num_points, epochs, batch_size, lr, grad_accum=1, loss_weights=None):
    ds = UnifiedMultiviewDataset(dset_path, split="train", num_points=num_points, image_size=224, augment=True)
    if len(ds) == 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir/"training_log.json").write_text(json.dumps([], indent=2))
        torch.save({"model": MultiViewPointCloudModel(num_points=num_points).state_dict()}, Path(out_dir)/"stage{}_epoch0.pth".format(stage))
        return str(out_dir)
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
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=mv_collate)
    dsv = UnifiedMultiviewDataset(dset_path, split="val", num_points=num_points, image_size=224, augment=False)
    dlv = DataLoader(dsv, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=mv_collate) if len(dsv) > 0 else None
    model = MultiViewPointCloudModel(num_points=num_points)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if stage == 2:
        for p in list(model.enc_front.parameters()) + list(model.enc_side.parameters()):
            p.requires_grad = False
    opt = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs,1))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log = []
    adjustments = []

    def _normalize_points(x):
        c = x.mean(dim=1, keepdim=True)
        y = x - c
        m = torch.norm(y, dim=2, keepdim=True).max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        y = y / m
        x1 = y[:, :, 0]
        y1 = y[:, :, 1]
        z1 = y[:, :, 2]
        out = torch.stack([x1, z1, -y1], dim=2)
        return out

    def _compute_scheduled_weights(ep_idx):
        base = loss_weights or {"chamfer":0.5,"fscore":0.2,"laplacian":0.1,"normal":0.15,"perceptual":0.05,"p2s":0.15,"silhouette":1.0}
        if ep_idx < 5:
            n_w = 0.5
        elif ep_idx < 10:
            n_w = 1.0
        elif ep_idx < 20:
            n_w = 2.0
        else:
            n_w = 3.0
        if ep_idx < 5:
            sil_w = base.get("silhouette", 1.0)
        elif ep_idx < 15:
            sil_w = 2.0
        else:
            sil_w = 3.0
        p2s_w = base.get("p2s", 0.15)
        if ep_idx >= 15:
            p2s_w = 0.10
        lap_w = base.get("laplacian", 0.1)
        if ep_idx >= 10 and ep_idx < 20:
            lap_w = max(lap_w, 0.2)
        elif ep_idx >= 20:
            lap_w = max(lap_w, 0.25)
        return {"chamfer":base["chamfer"],"fscore":base["fscore"],"laplacian":lap_w,"normal":n_w,"perceptual":base["perceptual"],"p2s":p2s_w,"silhouette":sil_w}

    def _evaluate_checkpoint(ep_idx, train_loss_avg):
        if dlv is None:
            return
        model.eval()
        with torch.no_grad():
            v_losses = []
            v_p2s = []
            v_norm = []
            v_comp = []
            v_sil_iou = []
            for batch in dlv:
                f = batch["front"].to(device)
                s = batch["side"].to(device)
                gt = batch["pointcloud"].to(device).permute(0,2,1)
                pts = model(f, s)
                w_ep = _compute_scheduled_weights(ep_idx)
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
                        h, w = int(batch["mask"].size(1)), int(batch["mask"].size(2))
                        _check_intrinsics_consistency("train", batch["camera_K"][0].detach().cpu().numpy(), h, w, f_scale=1.0)
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
                loss_val = w_ep["chamfer"]*cd2 + w_ep["fscore"]*f_loss + w_ep["laplacian"]*lap + w_ep["normal"]*nloss + w_ep["perceptual"]*ploss.mean() + w_ep.get("p2s",0.0)*p2s_val + w_ep.get("silhouette",0.1)*sil
                v_losses.append(loss_val.item())
                if isinstance(p2s_val, torch.Tensor):
                    v_p2s.append(float(p2s_val.item()))
                else:
                    v_p2s.append(float(p2s_val))
                v_norm.append(float(nloss.item()))
                try:
                    for i in range(pts.size(0)):
                        v_comp.append(mesh_completeness(pts[i].detach().cpu().numpy(), gt[i].detach().cpu().numpy()))
                except Exception:
                    pass
                try:
                    if isinstance(sil, torch.Tensor):
                        v_sil_iou.append(float(1.0 - sil.item()))
                    else:
                        v_sil_iou.append(float(1.0 - sil))
                except Exception:
                    pass
        metrics = {
            "epoch": int(ep_idx+1),
            "train_loss": float(train_loss_avg),
            "val_loss": float(np.mean(v_losses)) if len(v_losses) > 0 else 0.0,
            "p2s": float(np.mean(v_p2s)) if len(v_p2s) > 0 else 0.0,
            "normal_consistency": float(np.mean(v_norm)) if len(v_norm) > 0 else 0.0,
            "completeness": float(np.mean(v_comp)) if len(v_comp) > 0 else 0.0,
            "silhouette_iou": float(np.mean(v_sil_iou)) if len(v_sil_iou) > 0 else 0.0
        }
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir/ f"checkpoint_metrics_epoch{ep_idx+1}.json").write_text(json.dumps(metrics, indent=2))

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        accum = 0
        for batch in dl:
            f = batch["front"].to(device)
            s = batch["side"].to(device)
            gt = batch["pointcloud"].to(device).permute(0,2,1)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if stage == 1:
                    pts = model(f, f)
                else:
                    pts = model(f, s)
                pts = _normalize_points(pts)
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
                        h, w = int(batch["mask"].size(1)), int(batch["mask"].size(2))
                        _check_intrinsics_consistency("eval", batch["camera_K"][0].detach().cpu().numpy(), h, w, f_scale=1.0)
                        sil = silhouette_iou_loss(pts, batch["camera_K"], batch["mask"]).mean()
                    except Exception:
                        sil = 0.0
                w = _compute_scheduled_weights(ep)
                p2s = 0.0
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
                            p2s = torch.stack(vals).mean()
                    except Exception:
                        p2s = 0.0
                loss = w["chamfer"]*cd2 + w["fscore"]*f_loss + w["laplacian"]*lap + w["normal"]*nloss + w["perceptual"]*ploss.mean() + w.get("p2s",0.0)*p2s + w.get("silhouette",0.1)*sil
            scaler.scale(loss/grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            accum += 1
            if accum % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            ep_loss += loss.item()
        scheduler.step()
        avg = ep_loss/max(len(dl),1)
        log.append({"epoch": ep+1, "loss": avg})
        if len(log) >= 3:
            last3 = [d["loss"] for d in log[-3:]]
            if max(last3) - min(last3) < 1e-4:
                for g in opt.param_groups:
                    g['lr'] = g['lr'] * 0.5
                adjustments.append({"epoch": ep+1, "lr_factor": 0.5})
        if (ep+1) % 5 == 0:
            _evaluate_checkpoint(ep, avg)
        ck = Path(out_dir)/f"stage{stage}_epoch{ep+1}.pth"
        torch.save({"model": model.state_dict()}, ck)
    Path(out_dir)/"training_log.json"
    Path(out_dir/"training_log.json").write_text(json.dumps(log, indent=2))
    if len(adjustments) > 0:
        Path(out_dir/"adjustments.json").write_text(json.dumps(adjustments, indent=2))
    return str(out_dir)

def evaluate(dset_path, ckpt, num_points, out_dir):
    ds = UnifiedMultiviewDataset(dset_path, split="test", num_points=num_points, image_size=224, augment=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    model = MultiViewPointCloudModel(num_points=num_points)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"]) if "model" in state else model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    metrics = []
    dbg = Path(str(out_dir).replace("eval", "v4_debug"))
    dbg.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in dl:
            f = batch["front"].to(device)
            s = batch["side"].to(device)
            gt = batch["pointcloud"].to(device).permute(0,2,1)
            pts = model(f, s)
            cd = chamfer_distance_v2(pts, gt)
            bb = gt.max(dim=1)[0]-gt.min(dim=1)[0]
            diag = torch.norm(bb, dim=1)
            f1, f5, f10 = fscore_multi(pts, gt, diag)
            vi = voxel_iou(pts, gt, voxel_size=0.02)
            cats = batch.get("category", ["unknown"]*pts.size(0))
            if isinstance(cats, list):
                cat_list = cats
            else:
                try:
                    cat_list = list(cats)
                except Exception:
                    cat_list = ["unknown"]*pts.size(0)
            for i in range(pts.size(0)):
                metrics.append({"cd": float(cd[i].item()), "f1": float(f1[i].item()), "f5": float(f5[i].item()), "f10": float(f10[i].item()), "iou": float(vi[i].item()), "category": str(cat_list[i])})
            fp = dbg/"front.png"; sp = dbg/"side.png"
            try:
                import torchvision.utils as vutils
                vutils.save_image(f, str(fp))
                vutils.save_image(s, str(sp))
            except Exception:
                pass
            np.save(dbg/"pred_points.npy", pts.detach().cpu().numpy())
            np.save(dbg/"gt_points.npy", gt.detach().cpu().numpy())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir)/"metrics_summary.json"
    Path(out_dir/"metrics_summary.json").write_text(json.dumps({"items": metrics, "avg_cd": float(np.mean([m["cd"] for m in metrics])) if metrics else 0.0, "avg_f1": float(np.mean([m["f1"] for m in metrics])) if metrics else 0.0, "avg_f5": float(np.mean([m["f5"] for m in metrics])) if metrics else 0.0, "avg_f10": float(np.mean([m["f10"] for m in metrics])) if metrics else 0.0, "avg_iou": float(np.mean([m["iou"] for m in metrics])) if metrics else 0.0}, indent=2))
    return str(out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/checkpoints_multiview")
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--refined", type=int, default=4096)
    ap.add_argument("--epochs1", type=int, default=2)
    ap.add_argument("--epochs2", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--only-stage2", action="store_true")
    ap.add_argument("--resume2", type=str, default=None)
    args = ap.parse_args()
    out_root = Path(args.out)
    s1 = out_root/"stage1"
    s2 = out_root/"stage2"
    lw = None
    if args.config and Path(args.config).exists():
        try:
            cfg = json.loads(Path(args.config).read_text())
            lw = cfg.get("loss_weights", None)
        except Exception:
            lw = None
    if not args.only_stage2:
        train_stage(args.annotations, 1, s1, args.points, args.epochs1, args.batch, args.lr*2.0, grad_accum=2, loss_weights=lw)
        last1 = sorted(list(Path(s1).glob("*.pth")))[-1]
    # Stage 2 with optional resume
    if args.resume2 and Path(args.resume2).exists():
        try:
            base_state = torch.load(args.resume2, map_location="cpu")
            ds_tmp = UnifiedMultiviewDataset(args.annotations, split="train", num_points=args.points, image_size=224, augment=True)
            def mv_collate_tmp(batch):
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
            dl_tmp = DataLoader(ds_tmp, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=mv_collate_tmp)
            model_tmp = MultiViewPointCloudModel(num_points=args.points)
            model_tmp.load_state_dict(base_state["model"]) if "model" in base_state else model_tmp.load_state_dict(base_state)
            device_tmp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_tmp = model_tmp.to(device_tmp)
            # One forward to ensure shapes match
            try:
                b0 = next(iter(dl_tmp))
                f0 = b0["front"].to(device_tmp)
                s0 = b0.get("side", f0).to(device_tmp)
                _ = model_tmp(f0, s0)
            except Exception:
                pass
        except Exception:
            pass
    train_stage(args.annotations, 2, s2, args.points, args.epochs2, args.batch, args.lr, grad_accum=2, loss_weights=lw)
    last2 = sorted(list(Path(s2).glob("*.pth")))[-1]
    evaluate(args.annotations, str(last2), args.points, out_root/"eval")
    try:
        best_path = Path(s2)/"best.pth"
        if not best_path.exists():
            import shutil
            shutil.copy(str(last2), str(best_path))
    except Exception:
        pass

    try:
        log_path = Path(s2)/"training_log.json"
        if log_path.exists():
            data = json.loads(log_path.read_text())
            xs = [d.get("epoch", i+1) for i,d in enumerate(data)]
            ys = [d.get("loss", 0.0) for d in data]
            plt.figure()
            plt.plot(xs, ys, "-o")
            plt.title("Stage2 Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            out_curve = out_root/"loss_curve_stage2.png"
            plt.savefig(out_curve)
    except Exception:
        pass

    v4 = out_root/"v4"
    v4.mkdir(parents=True, exist_ok=True)
    try:
        mstate = torch.load(str(last2), map_location="cpu")
        sd = mstate["model"] if "model" in mstate else mstate
        torch.save(sd, str(v4/"model_v4.pth"))
        model = MultiViewPointCloudModel(num_points=args.points)
        model.load_state_dict(sd)
        torch.save(model.decoder.state_dict(), str(v4/"decoder_v4.pth"))
        torch.save({"enc_front": model.enc_front.state_dict(), "enc_side": model.enc_side.state_dict()}, str(v4/"encoder_v4.pth"))
        cfg = {"epochs1": args.epochs1, "epochs2": args.epochs2, "batch": args.batch, "points": args.points, "lr_stage1": args.lr*2.0, "lr_stage2": args.lr}
        (v4/"pipeline_config.json").write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass

if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.inference.infer import run_single
from src.evaluation.mesh_metrics import (
    mesh_point_to_surface_distance,
    mesh_edge_smoothness,
    mesh_completeness,
    mesh_normal_consistency,
)


def pick_pix3d_items(pix_root: Path, limit: int, category: str | None = None):
    meta = pix_root / "pix3d.json"
    if not meta.exists():
        return []
    data = json.loads(meta.read_text())
    out = []
    for it in data:
        if "img" in it and "model" in it:
            ip = pix_root / it["img"]
            mp = pix_root / it["model"]
            if category and it.get("category") != category:
                continue
            if ip.exists() and mp.exists():
                out.append({
                    "image": str(ip),
                    "gt_mesh": str(mp),
                    "category": it.get("category", "unknown"),
                })
        if len(out) >= limit:
            break
    return out


def compute_metrics(pred_mesh_path: Path, gt_mesh_path: Path):
    m_pred = o3d.io.read_triangle_mesh(str(pred_mesh_path))
    m_gt = o3d.io.read_triangle_mesh(str(gt_mesh_path))
    m_pred.compute_vertex_normals()
    m_gt.compute_vertex_normals()
    vp = np.asarray(m_pred.vertices)
    fp = np.asarray(m_pred.triangles)
    vg = np.asarray(m_gt.vertices)
    fg = np.asarray(m_gt.triangles)
    return {
        "point_to_surface": mesh_point_to_surface_distance(vg, fg, vp),
        "normal_consistency": mesh_normal_consistency(
            np.asarray(m_pred.vertex_normals), np.asarray(m_gt.vertex_normals)
        ),
        "edge_smoothness_std": mesh_edge_smoothness(vp, fp),
        "completeness": mesh_completeness(vp, vg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", type=str, default="results/mesh_eval_batch.json")
    ap.add_argument("--poisson-depth", type=int, default=10)
    ap.add_argument("--smooth-iter", type=int, default=15)
    ap.add_argument("--simplify", type=int, default=15000)
    ap.add_argument("--ckpt", type=str, default="results/checkpoints_multiview/stage2/best.pth")
    ap.add_argument("--use-model", action="store_true")
    ap.add_argument("--category", type=str, default=None)
    args = ap.parse_args()

    pix_root = Path("data/pix3d")
    items = pick_pix3d_items(pix_root, args.limit, args.category)
    results = []
    out_root = Path("results/infer_batch")
    out_root.mkdir(parents=True, exist_ok=True)

    for it in items:
        img = it["image"]
        gt = it["gt_mesh"]
        cat = it["category"]
        sub = out_root / Path(img).stem
        # Use 2048 points when loading model checkpoints to avoid mismatches
        p = run_single(
            img,
            args.ckpt if args.use_model else "checkpoints/dpt/best.pth",
            str(sub),
            refined=2048 if args.use_model else 4096,
            mesh=True,
            poisson_depth=args.poisson_depth,
            smooth_iter=args.smooth_iter,
            simplify=args.simplify,
            texture=True,
            use_model=args.use_model,
        )
        pred_mesh = Path(p) / "mesh.obj"
        if not pred_mesh.exists():
            continue
        mets = compute_metrics(pred_mesh, Path(gt))
        row = {
            "image": img,
            "gt_mesh": gt,
            "category": cat,
            "pred_dir": p,
        }
        row.update(mets)
        results.append(row)

    agg = {}
    if results:
        agg = {
            "count": len(results),
            "point_to_surface_mean": float(np.mean([r["point_to_surface"] for r in results])),
            "normal_consistency_mean": float(np.mean([r["normal_consistency"] for r in results])),
            "edge_smoothness_std_mean": float(np.mean([r["edge_smoothness_std"] for r in results])),
            "completeness_mean": float(np.mean([r["completeness"] for r in results])),
        }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"items": results, "aggregate": agg}, indent=2))
    try:
        import csv
        csv_p = out_p.with_suffix('.csv')
        with open(csv_p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["image","gt_mesh","category","point_to_surface","normal_consistency","edge_smoothness_std","completeness","pred_dir"]) 
            for r in results:
                w.writerow([r["image"], r["gt_mesh"], r["category"], r["point_to_surface"], r["normal_consistency"], r["edge_smoothness_std"], r["completeness"], r["pred_dir"]])
    except Exception:
        pass
    print(json.dumps(agg or {"count": 0}, indent=2))


if __name__ == "__main__":
    main()
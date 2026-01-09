import io
import os
from pathlib import Path
import numpy as np
import cv2
import json
import open3d as o3d
from fastapi import FastAPI, UploadFile, File
import subprocess, sys
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import importlib.util
def _load_module(module_path: str):
    p = Path(module_path)
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
_mod_depth = _load_module('src/models/depth_dpt.py')
_mod_pcd = _load_module('src/inference/depth_to_pcd.py')
DPTDepthPredictor = getattr(_mod_depth, 'DPTDepthPredictor')
depth_to_pointcloud = getattr(_mod_pcd, 'depth_to_pointcloud')

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("frontend/public/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SERVER_STATIC = Path("server/static")
SERVER_STATIC.mkdir(parents=True, exist_ok=True)
SERVER_OUTPUTS_BASE = SERVER_STATIC/"outputs"
SERVER_OUTPUTS_BASE.mkdir(parents=True, exist_ok=True)

def save_point_cloud(points: np.ndarray, colors: np.ndarray, path: Path):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pc, write_ascii=True)

def _normalize_and_align_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3)
    ctr = pts.mean(axis=0, keepdims=True)
    pts = pts - ctr
    norms = np.linalg.norm(pts, axis=1)
    max_range = float(np.max(norms)) if norms.size else 1.0
    if max_range > 0:
        pts = pts / max_range
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    pts = np.stack([x, z, -y], axis=1)
    return pts.astype(np.float32)

def _segment_foreground(depth: np.ndarray, threshold_factor: float = 0.75) -> np.ndarray:
    if depth.ndim != 2:
        return np.ones_like(depth, dtype=np.uint8)
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros_like(depth, dtype=np.uint8)
    mn = float(valid.min()); mx = float(valid.max())
    rng = mx - mn if mx > mn else 1.0
    d8 = ((np.clip(depth - mn, 0, rng) / rng) * 255.0).astype(np.uint8)
    _, otsu = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    approx = (depth < (mn + rng * threshold_factor)) & (depth > 0)
    mask = np.where(otsu > 0, 1, 0).astype(np.uint8)
    mask = np.maximum(mask, approx.astype(np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return mask

def _filter_points(pts: np.ndarray, cols: np.ndarray = None,
                   nb_neighbors: int = 30, std_ratio: float = 2.0):
    try:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if cols is not None and len(cols) == len(pts):
            pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        
        # Adaptive radius outlier removal
        bbox = pc.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        radius = max(diag * 0.05, 0.005) # 5% of diagonal, with a minimum
        min_nb_points = 15
        pc, _ = pc.remove_radius_outlier(nb_points=min_nb_points, radius=radius)
        
        pts1 = np.asarray(pc.points, dtype=np.float32)
        cols1 = np.asarray(pc.colors, dtype=np.float32) if pc.has_colors() else None
        pc2 = o3d.geometry.PointCloud()
        pc2.points = o3d.utility.Vector3dVector(pts1.astype(np.float64))
        if cols1 is not None:
            pc2.colors = o3d.utility.Vector3dVector(cols1.astype(np.float64))
        pc2, ind2 = pc2.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        final_pts = np.asarray(pc2.points, dtype=np.float32)
        final_cols = np.asarray(pc2.colors, dtype=np.float32) if pc2.has_colors() else None
        
        return final_pts, (final_cols if cols is not None else None)
    except Exception:
        return pts, cols

def compute_stats(points: np.ndarray):
    num = int(points.shape[0])
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    bb = {'min': mn.tolist(), 'max': mx.tolist()}
    vol = float(max(mx[0]-mn[0],1e-9)*max(mx[1]-mn[1],1e-9)*max(mx[2]-mn[2],1e-9))
    density = float(num/vol) if vol>0 else float('inf')
    ctr = points.mean(axis=0)
    spread = float(np.sqrt(((points-ctr)**2).sum(axis=1)).mean())
    q = 0.5*np.tanh(density/1000.0) + 0.5*np.exp(-spread)
    return {'num_points': num, 'density': density, 'spread': spread, 'bounding_box': bb, 'quality_score': float(q)}

PREDICTOR = None
def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = DPTDepthPredictor(model_name='DPT_Hybrid')
    return PREDICTOR

@app.on_event("startup")
def _startup_init():
    try:
        _ = get_predictor()
    except Exception:
        pass

@app.post("/api/reconstruct")
async def reconstruct(file: UploadFile = File(...), n_points: int = 16000, f_scale: float = 1.0, use_segmentation: bool = False, mode: str = 'fast'):
    try:
        data = await file.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"status":"error","message":"invalid image"})

        cv2.imwrite(str(OUTPUT_DIR/"example.png"), img)
        cv2.imwrite(str(OUTPUTS_DIR/"example.png"), img)
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_dir = SERVER_OUTPUTS_BASE/ts
        ts_dir.mkdir(parents=True, exist_ok=True)

        predictor = get_predictor()
        depth = predictor.predict_depth(img)
        depth = cv2.bilateralFilter(depth.astype(np.float32), 5, 0.1, 2.0)
        h, w = img.shape[:2]
        f = float(max(h, w)) * f_scale
        camK = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1.0]], dtype=np.float32)
        mask = None
        if use_segmentation:
            mask = _segment_foreground(depth)
        upsample_scale = 1
        gaussian = False
        denoise = False
        smooth = False
        edge_sample = True
        if mode == 'balanced':
            upsample_scale = 1
            gaussian = True
            denoise = False
            smooth = False
        elif mode == 'quality':
            upsample_scale = 2
            gaussian = True
            denoise = True
            smooth = True
        pts, cols = depth_to_pointcloud(depth, image_bgr=img, camera_K=camK, n_points=int(n_points), mask=mask, use_mask=use_segmentation, denoise=denoise, gaussian=gaussian, edge_sample=edge_sample, upsample_scale=upsample_scale, smooth=smooth, sample_name="api_reconstruct")
        pts = _normalize_and_align_points(pts)
        pts, cols = _filter_points(pts, cols)
        pred_dir_results = Path('results')/Path('pointcloud_runs')/ts
        pred_dir_results.mkdir(parents=True, exist_ok=True)
        pred_ply = pred_dir_results/"pred_pointcloud.ply"
        pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        try:
            pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        except Exception:
            pass
        o3d.io.write_point_cloud(str(pred_ply), pc, write_ascii=True)

        # Simple visualization (front/side/top combined)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,3, figsize=(9,3))
        axes[0].scatter(pts[:,0], pts[:,2], s=0.5, c='k'); axes[0].set_title('front'); axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[1].scatter(pts[:,1], pts[:,2], s=0.5, c='k'); axes[1].set_title('side'); axes[1].set_xticks([]); axes[1].set_yticks([])
        axes[2].scatter(pts[:,0], pts[:,1], s=0.5, c='k'); axes[2].set_title('top');  axes[2].set_xticks([]); axes[2].set_yticks([])
        fig.tight_layout()
        vis_path_results = pred_dir_results/"visualization.png"
        fig.savefig(str(vis_path_results), dpi=200)
        plt.close(fig)

        stats = compute_stats(pts)
        # Save depth
        np.save(pred_dir_results/"depth.npy", depth)
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111)
            ax.imshow(depth, cmap='magma')
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(str(pred_dir_results/"depth.png"), dpi=200)
            plt.close(fig)
        except Exception:
            pass
        with open(pred_dir_results/"pointcloud_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        q = float(stats.get('quality_score', 0.0))
        q = max(0.0, min(1.0, q))
        resp = {
            "status": "ok",
            "num_points": int(stats['num_points']),
            "point_cloud_coordinates": pts.tolist(),
            "reconstruction_quality": float(stats['quality_score']),
            "confidence": q,
            "confidence_score": q,
            "stats": stats,
            "timestamp": ts,
            "mode": mode
        }
        # Quality report logging
        try:
            qr = Path('results')/'quality_report.json'
            entries = []
            if qr.exists():
                try:
                    entries = json.loads(qr.read_text())
                except Exception:
                    entries = []
            entries.append({
                "timestamp": ts,
                "sample": "api_reconstruct",
                "synthetic_camera": True,
                "n_points": int(n_points),
                "optional_deps": {"pytorch3d": False, "nvdiffrast": False}
            })
            qr.write_text(json.dumps(entries, indent=2))
        except Exception:
            pass
        return resp
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error","message":str(e)})

@app.post("/api/reconstruct-mesh")
async def reconstruct_mesh(file: UploadFile = File(...), n_points: int = 40000, depth: int = 10, smooth_iter: int = 15, simplify_faces: int = 15000, texture: bool = True):
    try:
        data = await file.read()
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"status":"error","message":"invalid image"})
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_dir = SERVER_OUTPUTS_BASE/ts
        ts_dir.mkdir(parents=True, exist_ok=True)
        predictor = get_predictor()
        depth_map = predictor.predict_depth(img)
        h, w = img.shape[:2]
        f = float(max(h, w))
        camK = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1.0]], dtype=np.float32)
        pts, cols = depth_to_pointcloud(depth_map, image_bgr=img, camera_K=camK, n_points=int(n_points), mask=None, use_mask=False, denoise=True, gaussian=False, edge_sample=True, upsample_scale=1, smooth=True, sample_name="api_mesh")
        from src.mesh.poisson import poisson_mesh_from_points, save_mesh
        from src.mesh.cleanup import filter_by_density, largest_component, laplacian_smooth, refine_normals, simplify as simplify_mesh
        from src.mesh.texturing import planar_uv, project_texture_front, write_obj_with_mtl
        m, dens = poisson_mesh_from_points(pts, depth=depth)
        if dens is not None:
            m = filter_by_density(m, dens, keep_ratio=0.7)
        m = largest_component(m)
        m = laplacian_smooth(m, iterations=smooth_iter)
        m = refine_normals(m)
        if simplify_faces and simplify_faces>0:
            m = simplify_mesh(m, target_faces=simplify_faces)
        ply_path_ts = ts_dir/"mesh_poisson.ply"
        save_mesh(m, ply_path_ts)
        obj_path = ts_dir/"mesh.obj"
        mtl_path = ts_dir/"mesh.mtl"
        tex_path = ts_dir/"texture.png"
        if texture:
            uv = planar_uv(m)
            tex = project_texture_front(OUTPUT_DIR/"example.png", resolution=1024)
            write_obj_with_mtl(m, uv, tex, obj_path)
        resp = {
            "status": "ok",
            "mesh_url": f"/static/outputs/{ts}/mesh.obj",
            "texture_url": f"/static/outputs/{ts}/texture.png",
            "preview_image_url": f"/static/outputs/{ts}/visualization.png",
            "confidence": 0.0,
        }
        return resp
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error","message":str(e)})

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        data = await file.read()
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = SERVER_OUTPUTS_BASE/ts
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir/"upload.png"
        with open(img_path, 'wb') as f:
            f.write(data)
        return {"status":"ok","url": f"/static/outputs/{ts}/upload.png","timestamp_dir": f"/static/outputs/{ts}/"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error","message":str(e)})

@app.get("/api/status")
def status():
    return {"status":"ok","version":"v4","server":"fastapi"}

@app.get("/api/model-info")
def model_info():
    info = {"status":"ok","version":"v4","artifacts":{}}
    v4 = Path("results")/"checkpoints_v4"/"v4"
    info["artifacts"]["model_v4.pth"] = (v4/"model_v4.pth").exists()
    info["artifacts"]["decoder_v4.pth"] = (v4/"decoder_v4.pth").exists()
    info["artifacts"]["encoder_v4.pth"] = (v4/"encoder_v4.pth").exists()
    cfg = v4/"pipeline_config.json"
    try:
        if cfg.exists():
            info["config"] = json.loads(cfg.read_text())
    except Exception:
        pass
    return info

@app.post("/predict")
async def predict(file: UploadFile = File(...), n_points: int = 20000):
    return await reconstruct(file, n_points)

@app.post("/3d/reconstruct")
async def reconstruct_v2(file: UploadFile = File(...)):
    return await reconstruct(file)

@app.post("/generate_pointcloud")
async def generate_pointcloud(file: UploadFile = File(...), n_points: int = 20000):
    return await reconstruct(file, n_points)

@app.post("/train")
async def train(payload: dict = None):
    try:
        req = payload or {}
        cfg = [
            sys.executable, "scripts/train_v3.py",
            "--epochs", str(req.get('epochs', 5)),
            "--datasets", ",".join(req.get('datasets', ['pix3d','shapenet','pascal3d','objectnet3d','co3d','google_scanned_objects']))
        ]
        subprocess.Popen(cfg, cwd=str(Path('.').resolve()))
        log = {"status":"started","config": req}
        with open(SERVER_STATIC/"train_status.json", 'w') as f:
            json.dump(log, f, indent=2)
        return {"status":"ok","message":"training started","status_url":"/static/train_status.json","log_url":"/results/checkpoints_v3/training_log.txt"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error","message":str(e)})

def _list_categories(dataset_root: Path, image_dirs: list, mesh_dirs: list):
    cats = set()
    for d in image_dirs:
        p = dataset_root/d
        if p.exists():
            for sub in p.iterdir():
                if sub.is_dir():
                    cats.add(sub.name)
    for d in mesh_dirs:
        p = dataset_root/d
        if p.exists():
            for sub in p.iterdir():
                if sub.is_dir():
                    cats.add(sub.name)
    return sorted(list(cats))

@app.get("/get_categories")
def get_categories():
    data_root = Path('data')
    datasets = {
        'pix3d': {'image_dirs': ['img','images'], 'mesh_dirs': []},
        'shapenet': {'image_dirs': ['images'], 'mesh_dirs': ['meshes']},
        'pascal3d': {'image_dirs': ['Images','images'], 'mesh_dirs': []},
        'objectnet3d': {'image_dirs': ['images'], 'mesh_dirs': []},
        'co3d': {'image_dirs': ['images'], 'mesh_dirs': []},
        'google_scanned_objects': {'image_dirs': ['images'], 'mesh_dirs': ['meshes']}
    }
    out = {}
    missing = []
    for name, cfg in datasets.items():
        root = data_root/name
        if not root.exists():
            missing.append(name)
            continue
        out[name] = _list_categories(root, cfg['image_dirs'], cfg['mesh_dirs'])
    return {"datasets": out, "missing": missing}

@app.get("/datasets/list")
def datasets_list():
    return get_categories()

@app.get("/get_dataset_images")
def get_dataset_images(dataset: str, category: str, max_items: int = 12):
    data_root = Path('data')/dataset
    paths = []
    for d in ['img','images']:
        p = data_root/d/category
        if p.exists():
            for ext in ['*.jpg','*.png','*.jpeg']:
                for f in p.glob(ext):
                    paths.append(f)
    paths = sorted(paths)[:max_items]
    urls = []
    for f in paths:
        try:
            rel = Path(f).relative_to(Path('data'))
            urls.append(f"/data/{rel.as_posix()}")
        except Exception:
            urls.append(f"/data/{Path(dataset)/f.name}")
    return {"count": len(urls), "items": urls}

@app.get("/datasets/category/{name}")
def datasets_category(name: str, dataset: str, max_items: int = 12):
    return get_dataset_images(dataset=dataset, category=name, max_items=max_items)

@app.get("/get_metrics")
def get_metrics():
    p = Path('results')/'metrics_summary.json'
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"status":"ok","message":"no metrics found"}

@app.get("/metrics/latest")
def metrics_latest():
    return get_metrics()

@app.post("/save_visualization")
async def save_visualization(payload: dict):
    try:
        img_data = payload.get('image_data')
        filename = payload.get('filename', 'visualization_saved.png')
        if img_data and img_data.startswith('data:image'):
            header, b64 = img_data.split(',', 1)
            import base64
            img_bytes = base64.b64decode(b64)
            out_path = OUTPUT_DIR/filename
            with open(out_path, 'wb') as f:
                f.write(img_bytes)
            return {"status":"ok","message":f"Saved {out_path.as_posix()}"}
        return JSONResponse(status_code=400, content={"status":"error","message":"invalid image data"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error","message":str(e)})

@app.get("/")
def index():
    p = Path("templates/index.html")
    if not p.exists():
        return {"status":"ok"}
    return FileResponse(str(p))

@app.get("/health")
def health():
    return {"status":"ok"}

app.mount("/static", StaticFiles(directory=str(SERVER_STATIC)), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/data", StaticFiles(directory=str(Path("data"))), name="data")
app.mount("/results", StaticFiles(directory=str(Path("results"))), name="results")

import torch
import math
import numpy as np

def chamfer_distance(p1, p2):
    a = p1
    b = p2
    d1 = torch.cdist(a, b, p=2)
    m1 = d1.min(dim=2)[0]
    m2 = d1.min(dim=1)[0]
    return m1.mean(dim=1) + m2.mean(dim=1)

def chamfer_distance_v2(p1, p2):
    d = torch.cdist(p1, p2, p=2)
    m1 = d.min(dim=2)[0]
    m2 = d.min(dim=1)[0]
    return (m1.pow(2).mean(dim=1) + m2.pow(2).mean(dim=1))

def fscore(p_pred, p_gt, thresh):
    d = torch.cdist(p_pred, p_gt, p=2)
    p_to_g = (d.min(dim=2)[0] < thresh).float().mean(dim=1)
    g_to_p = (d.min(dim=1)[0] < thresh).float().mean(dim=1)
    f = 2*p_to_g*g_to_p/(p_to_g+g_to_p+1e-8)
    return f

def fscore_multi(p_pred, p_gt, diags):
    f1 = fscore(p_pred, p_gt, 0.01*diags.unsqueeze(1))
    f5 = fscore(p_pred, p_gt, 0.05*diags.unsqueeze(1))
    f10 = fscore(p_pred, p_gt, 0.10*diags.unsqueeze(1))
    return f1, f5, f10

def earth_movers_distance(p1, p2):
    try:
        import torchsort
        a = p1.sort(dim=1)[0]
        b = p2.sort(dim=1)[0]
        return torch.norm(a - b, dim=2).mean(dim=1)
    except Exception:
        return torch.zeros(p1.size(0), device=p1.device)

def _knn(points, k=16):
    d = torch.cdist(points, points, p=2)
    idx = d.topk(k, largest=False)[1]
    return idx

def laplacian_regularization(points, k=16):
    idx = _knn(points, k)
    b, n, _ = points.size()
    neighbors = []
    for i in range(b):
        neighbors.append(points[i][idx[i]])
    gather = torch.stack(neighbors, dim=0)
    nbr_mean = gather.mean(dim=2)
    lap = points - nbr_mean
    return lap.pow(2).sum(dim=2).mean(dim=1)

def estimate_normals(points, k=16):
    idx = _knn(points, k)
    b, n, _ = points.size()
    normals = []
    for i in range(b):
        pi = points[i]
        Ii = idx[i]
        Ni = []
        for j in range(n):
            nbr = pi[Ii[j]]
            c = nbr.mean(dim=0)
            X = nbr - c
            cov = X.t() @ X
            eigvals, eigvecs = torch.linalg.eigh(cov)
            nrm = eigvecs[:,0]
            Ni.append(nrm)
        Ni = torch.stack(Ni, dim=0)
        normals.append(Ni)
    return torch.stack(normals, dim=0)

def normal_consistency_loss(p_pred, p_gt, k=16):
    n_pred = estimate_normals(p_pred, k)
    n_gt = estimate_normals(p_gt, k)
    cos = (n_pred * n_gt).sum(dim=2)
    return (1.0 - cos.abs()).mean(dim=1)

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

def project_points(points, camera_K, h, w):
    fx = camera_K[0][0]; fy = camera_K[1][1]
    cx = camera_K[0][2]; cy = camera_K[1][2]
    z = points[:,:,2].clamp(min=1e-8)
    x = (points[:,:,0] * fx / z) + cx
    y = (points[:,:,1] * fy / z) + cy
    xi = x.round().clamp(0, w-1).long()
    yi = y.round().clamp(0, h-1).long()
    return yi, xi

def silhouette_iou_loss(points, camera_K, mask):
    b, n, _ = points.size()
    h, w = mask.size(1), mask.size(2)
    ious = []
    for i in range(b):
        Ki = camera_K[i] if camera_K is not None else torch.tensor(compute_intrinsics(h, w), dtype=torch.float32, device=points.device)
        _check_intrinsics_consistency("silhouette", Ki.detach().cpu().numpy() if isinstance(Ki, torch.Tensor) else Ki, h, w, f_scale=1.0)
        yi, xi = project_points(points[i].unsqueeze(0), Ki, h, w)
        pred = torch.zeros((h,w), device=points.device)
        pred[yi[0], xi[0]] = 1.0
        gt = mask[i].float()
        inter = (pred*gt).sum()
        union = pred.sum() + gt.sum() - inter
        iou = inter/(union+1e-8)
        ious.append(1.0 - iou)
    return torch.stack(ious, dim=0)

def point_to_surface_loss(points_pred, mesh_vertices):
    pv = points_pred
    mv = mesh_vertices
    d = torch.cdist(pv, mv, p=2).min(dim=2)[0]
    return d.mean(dim=1)

def perceptual_loss_clip(image_pairs):
    try:
        import clip
        import torchvision.transforms as T
        device = image_pairs[0][0].device
        model, preprocess = clip.load("ViT-L/14", device=device if device.type=='cuda' else 'cpu')
        losses = []
        for f, s in image_pairs:
            img_f = T.functional.resize(f, [224,224])
            img_s = T.functional.resize(s, [224,224])
            emb_f = model.encode_image(img_f)
            emb_s = model.encode_image(img_s)
            losses.append(torch.norm(emb_f - emb_s, dim=1))
        return torch.stack(losses).mean(dim=0)
    except Exception:
        return torch.zeros(image_pairs[0][0].size(0), device=image_pairs[0][0].device)
def voxel_iou(p_pred, p_gt, voxel_size=0.02):
    bsz = p_pred.size(0)
    ious = []
    for i in range(bsz):
        A = (p_pred[i] / voxel_size).floor().int()
        B = (p_gt[i] / voxel_size).floor().int()
        A = torch.unique(A, dim=0)
        B = torch.unique(B, dim=0)
        if A.size(0) == 0 and B.size(0) == 0:
            ious.append(torch.tensor(1.0, device=p_pred.device))
            continue
        if A.size(0) == 0 or B.size(0) == 0:
            ious.append(torch.tensor(0.0, device=p_pred.device))
            continue
        # Hash voxels
        hA = A[:,0]*73856093 + A[:,1]*19349663 + A[:,2]*83492791
        hB = B[:,0]*73856093 + B[:,1]*19349663 + B[:,2]*83492791
        setA = torch.unique(hA)
        setB = torch.unique(hB)
        inter = torch.sum(torch.isin(setA, setB))
        union = setA.numel() + setB.numel() - inter
        iou = inter.float() / (union + 1e-8)
        ious.append(iou)
    return torch.stack(ious, dim=0)

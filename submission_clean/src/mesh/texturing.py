import numpy as np
from pathlib import Path
from PIL import Image

def planar_uv(mesh):
    try:
        import open3d as o3d
        v = np.asarray(mesh.vertices)
        mn = v.min(axis=0); mx = v.max(axis=0)
        uv = (v[:, :2] - mn[:2]) / np.maximum(mx[:2]-mn[:2], 1e-8)
        return uv
    except Exception:
        return None

def project_texture_front(image_path, resolution=1024):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((resolution, resolution))
    return img

def write_obj_with_mtl(mesh, uv, texture_img, out_obj):
    out_obj = Path(out_obj)
    out_mtl = out_obj.with_suffix('.mtl')
    tex_path = out_obj.parent / 'texture.png'
    texture_img.save(tex_path)
    try:
        import open3d as o3d
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        with open(out_obj, 'w') as fo:
            fo.write(f"mtllib {out_mtl.name}\n")
            for i in range(v.shape[0]):
                fo.write(f"v {v[i,0]} {v[i,1]} {v[i,2]}\n")
            for i in range(uv.shape[0]):
                fo.write(f"vt {uv[i,0]} {uv[i,1]}\n")
            fo.write("usemtl material_0\n")
            for i in range(f.shape[0]):
                a,b,c = f[i]
                fo.write(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n")
        with open(out_mtl, 'w') as fm:
            fm.write("newmtl material_0\n")
            fm.write("Kd 1.000 1.000 1.000\n")
            fm.write(f"map_Kd {tex_path.name}\n")
        return str(out_obj), str(out_mtl), str(tex_path)
    except Exception:
        return str(out_obj), str(out_mtl), str(tex_path)
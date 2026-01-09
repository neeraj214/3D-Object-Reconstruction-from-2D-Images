import requests
import sys
import os
from pathlib import Path

def save_to_results(base_url: str, outputs: dict, out_root: Path):
    ts_dir = outputs.get('timestamp_dir') or ''
    ts = ts_dir.strip('/').split('/')[-1] if ts_dir else 'run'
    out_dir = out_root/ts
    out_dir.mkdir(parents=True, exist_ok=True)
    def fetch(rel, name):
        if not rel:
            return None
        url = base_url + rel
        r = requests.get(url)
        if r.status_code == 200:
            p = out_dir/name
            with open(p, 'wb') as f:
                f.write(r.content)
            return str(p)
        return None
    pc = fetch(outputs.get('timestamp_point_cloud') or outputs.get('point_cloud'), 'point_cloud.ply')
    viz = fetch(outputs.get('timestamp_visualization') or outputs.get('visualization'), 'visualization.png')
    stats = fetch(outputs.get('timestamp_stats') or outputs.get('stats'), 'pointcloud_stats.json')
    depth = fetch(outputs.get('depth_png'), 'depth.png')
    return {'dir': str(out_dir), 'point_cloud': pc, 'visualization': viz, 'stats': stats, 'depth': depth}

def main(path, save=False):
    base = 'http://127.0.0.1:5000'
    url = base + '/predict'
    with open(path, 'rb') as f:
        r = requests.post(url, files={'file': f})
    print(r.status_code)
    try:
        j = r.json()
        print('predicted_class:', j.get('predicted_class'))
        print('confidence_score:', j.get('confidence_score'))
        print('points_len:', len(j.get('point_cloud_coordinates', [])))
        ou = j.get('output_urls', {})
        print('timestamp_point_cloud:', ou.get('timestamp_point_cloud'))
        print('timestamp_visualization:', ou.get('timestamp_visualization'))
        print('stats_url:', ou.get('timestamp_stats'))
        print('reconstruction_quality:', j.get('reconstruction_quality'))
        if save:
            ts = j.get('timestamp') or 'run'
            out_root = Path('results')/Path('pointcloud_runs')/ts
            out_root.mkdir(parents=True, exist_ok=True)
            # server-side save may exist
            server_ply = out_root/'pred_pointcloud.ply'
            local_ply = out_root/'pred_pointcloud_client.ply'
            # write local ply from coordinates if server file missing
            if not server_ply.exists():
                try:
                    import open3d as o3d
                    import numpy as np
                    arr = np.array(j.get('point_cloud_coordinates', []), dtype=np.float32)
                    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(arr.astype(np.float64))
                    o3d.io.write_point_cloud(str(local_ply), pc, write_ascii=True)
                except Exception:
                    pass
            print('saved_dir:', str(out_root))
            print('saved_point_cloud:', str(server_ply if server_ply.exists() else local_ply if local_ply.exists() else 'None'))
    except Exception:
        print(r.text[:1000])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/test_predict.py <image_path> [--save]')
        raise SystemExit(1)
    img = sys.argv[1]
    save_flag = len(sys.argv) > 2 and sys.argv[2] == '--save'
    main(img, save_flag)
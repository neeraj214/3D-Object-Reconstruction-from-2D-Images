import os
import sys
import tarfile
import zipfile
import requests
from pathlib import Path

def _ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def _download(url, out_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return out_path

def _extract(archive_path, out_dir):
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(out_dir)
    elif archive_path.endswith('.tar') or archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path) as t:
            t.extractall(out_dir)
    else:
        return False
    return True

def download_pix3d(output_dir='data/pix3d/'):
    _ensure_dir(output_dir)
    meta = Path(output_dir) / 'README_download.txt'
    with open(meta, 'w') as f:
        f.write('Pix3D requires manual download from the official source. Place files under this directory.')
    return str(output_dir)

def download_shapenet_subset(output_dir='data/shapenet/', categories=None):
    _ensure_dir(output_dir)
    cats = categories or ['chair','car','table']
    placeholder = Path(output_dir) / 'README_download.txt'
    with open(placeholder, 'w') as f:
        f.write('ShapeNet subset requires manual download. Expected categories: ' + ','.join(cats))
    return str(output_dir)

def download_pascal3d(output_dir='data/pascal3d/'):
    _ensure_dir(output_dir)
    url = 'http://cvgl.stanford.edu/projects/pascal3d/PASCAL3D+_release1.1.zip'
    archive = Path(output_dir) / 'PASCAL3D+_release1.1.zip'
    try:
        _download(url, archive)
        _extract(str(archive), output_dir)
        return str(output_dir)
    except Exception as e:
        with open(Path(output_dir)/'README_download.txt', 'w') as f:
            f.write(f'Automatic download failed. Error: {e}\nVisit: {url}')
        return str(output_dir)

def download_objectnet3d(output_dir='data/objectnet3d/'):
    _ensure_dir(output_dir)
    with open(Path(output_dir)/'README_download.txt', 'w') as f:
        f.write('ObjectNet3D requires manual access. Place extracted data here.')
    return str(output_dir)

def download_co3d(output_dir='data/co3d/'):
    _ensure_dir(output_dir)
    with open(Path(output_dir)/'README_download.txt', 'w') as f:
        f.write('CO3D is large and requires manual download. See: https://github.com/facebookresearch/co3d')
    return str(output_dir)

def download_google_scanned(output_dir='data/google_scanned/'):
    _ensure_dir(output_dir)
    with open(Path(output_dir)/'README_download.txt', 'w') as f:
        f.write('Google Scanned Objects: place dataset here. Reference: https://research.google/tools/datasets/')
    return str(output_dir)

def write_dataset_info(dataset_root, name, source_url=None):
    info = {
        'name': name,
        'root': str(dataset_root),
        'source_url': source_url or '',
        'format': {
            'images': 'images/',
            'masks': 'masks/',
            'annotations': 'annotations.json',
            'pointclouds': 'pointclouds/'
        }
    }
    with open(Path(dataset_root)/'dataset_info.json','w') as f:
        import json
        json.dump(info, f, indent=2)
    # ensure subdirs
    for d in ['images','masks','pointclouds']:
        _ensure_dir(Path(dataset_root)/d)
    ann = Path(dataset_root)/'annotations.json'
    if not ann.exists():
        with open(ann,'w') as f:
            f.write('[]')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'pix3d':
        print(download_pix3d())
    elif cmd == 'shapenet':
        print(download_shapenet_subset())
    elif cmd == 'pascal3d':
        p = download_pascal3d()
        write_dataset_info(p, 'Pascal3D+', 'http://cvgl.stanford.edu/projects/pascal3d/')
        print(p)
    elif cmd == 'objectnet3d':
        p = download_objectnet3d()
        write_dataset_info(p, 'ObjectNet3D', 'http://cvgl.stanford.edu/projects/objectnet3d/')
        print(p)
    elif cmd == 'co3d':
        p = download_co3d()
        write_dataset_info(p, 'CO3D', 'https://github.com/facebookresearch/co3d')
        print(p)
    elif cmd == 'google_scanned':
        p = download_google_scanned()
        write_dataset_info(p, 'Google Scanned Objects', 'https://research.google/tools/datasets/')
        print(p)
    else:
        print('Usage: python scripts/data_download.py [pix3d|shapenet|pascal3d|objectnet3d|co3d|google_scanned]')
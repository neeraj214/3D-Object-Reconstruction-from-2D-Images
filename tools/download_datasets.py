import os
import sys
import argparse
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    parser = argparse.ArgumentParser(description="Download and setup datasets")
    parser.add_argument("--dataset", type=str, choices=['pix3d', 'shapenet'], default='pix3d')
    parser.add_argument("--out_dir", type=str, default='data')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == 'pix3d':
        url = "http://pix3d.csail.mit.edu/data/pix3d.zip"
        zip_path = os.path.join(args.out_dir, "pix3d.zip")
        
        if not os.path.exists(os.path.join(args.out_dir, "pix3d")):
            if not os.path.exists(zip_path):
                print(f"Downloading Pix3D from {url}...")
                download_file(url, zip_path)
            
            unzip_file(zip_path, args.out_dir)
            
            # Cleanup zip if needed (optional)
            # os.remove(zip_path)
        else:
            print("Pix3D directory already exists.")
            
    elif args.dataset == 'shapenet':
        print("ShapeNet requires an account to download.")
        print("Please register at https://shapenet.org/ and download ShapeNetCore.v2.")
        print(f"Extract it to {os.path.join(args.out_dir, 'ShapeNetCore.v2')}")

if __name__ == "__main__":
    main()

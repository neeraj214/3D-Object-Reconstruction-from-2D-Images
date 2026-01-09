import os
import torch
import numpy as np
import cv2

def _get_transform_offline():
    try:
        tf = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        print("[INFO] Using MiDaS official transforms.")
        return tf, 'midas'
    except Exception as e:
        from torchvision import transforms
        print("[WARN] MiDaS transforms failed. Using offline fallback. Error:", e)
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]), 'compose'

class DPTDepthPredictor:
    def __init__(self, device=None, model_name='DPT_Hybrid', cache_dir=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'intel-isl_MiDaS_master')
        ckpt_map = {
            'DPT_Hybrid': 'dpt_hybrid_384.pt',
            'DPT_Large': 'dpt_large_384.pt'
        }
        ckpt_name = ckpt_map.get(model_name, None)
        model = None
        if ckpt_name is not None:
            cache_path = os.path.expanduser(os.path.join('~', '.cache', 'torch', 'hub', 'checkpoints', ckpt_name))
            if os.path.exists(cache_path):
                print("[INFO] Using cached MiDaS model:", cache_path)
                try:
                    state = torch.load(cache_path, map_location=self.device)
                    model = torch.hub.load('intel-isl/MiDaS', model_name, trust_repo=True, skip_validation=True)
                    model.load_state_dict(state)
                except Exception:
                    model = None
        if model is None:
            try:
                model = torch.hub.load('intel-isl/MiDaS', model_name, trust_repo=True, skip_validation=True)
            except Exception:
                model = torch.hub.load(self.cache_dir, model_name)
        self.model = model.to(self.device)
        self.model.eval()
        tf, tf_type = _get_transform_offline()
        self._tf_type = tf_type
        self.tf = tf

    def predict_depth(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self._tf_type == 'compose':
            x = self.tf(torch.from_numpy(img_rgb).permute(2,0,1).float()/255.0)
            if x.dim() == 3:
                x = x.unsqueeze(0)
        else:
            x = self.tf(img_rgb)
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.model(x)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=img_rgb.shape[:2], mode='bicubic', align_corners=False
            ).squeeze()
        d = pred.detach().cpu().numpy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        return d
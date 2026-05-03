"""
clinical_review.py — Công cụ hậu kiểm lâm sàng dành cho bác sĩ.

Tính năng:
    1. Tự động tìm kiếm 3 ca tiêu biểu: Chắc chắn có LVO, chắc chắn có Lesion, chắc chắn có CoW.
    2. Hiển thị 2 ảnh so sánh duy nhất: GT (Bác sĩ) vs Pred (AI).
    3. Chồng cả 3 nhãn lên cùng một ảnh não gốc CTA.
"""

import os
import sys
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse

# Tự động thêm thư mục 'src' vào hệ thống để import module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.dual_unet import build_model
from data.dataset import ISLES24Dataset
from evaluation.visualize import overlay_predictions

def get_clinical_samples(data_dir, num_checks=500):
    """Tìm 3 file npy thỏa mãn điều kiện có LVO, Lesion, CoW"""
    all_files = glob(os.path.join(data_dir, "*.npy"))
    np.random.shuffle(all_files)
    
    found = {"lvo": None, "lesion": None, "cow": None}
    
    print(f"[Review] Đang quét {num_checks} mẫu để tìm ca tiêu biểu...")
    for f in all_files[:num_checks]:
        # Chúng ta dùng Dataset để nạp dữ liệu để kích hoạt Gaussian Heatmap tự động
        temp_ds = ISLES24Dataset([f], transform=None)
        sample = temp_ds[0]
        lbl = sample["label"] # (3, H, W) - Đã là heatmap cho LVO
        
        # Ưu tiên ca có LVO đủ rõ để bác sĩ thấy (Heatmap sum > 80)
        if found["lvo"] is None and lbl[1].sum() > 80: 
            found["lvo"] = f
        if found["lesion"] is None and lbl[0].sum() > 300: # Lấy ca Lesion to hơn
            found["lesion"] = f
        if found["cow"] is None and lbl[2].sum() > 200: 
            found["cow"] = f
            
        if all(v is not None for v in found.values()):
            break
            
    return found


def run_clinical_review(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Config & Model
    # Logic nạp config linh hoạt (Hỗ trợ Kaggle/Local)
    def load_cfg(name):
        paths = [f"src/configs/{name}.yaml", f"configs/{name}.yaml", f"../configs/{name}.yaml"]
        for p in paths:
            if os.path.exists(p):
                with open(p, "r") as f: return yaml.safe_load(f)
        raise FileNotFoundError(f"Không tìm thấy file cấu hình {name}.yaml tại {paths}")

    model_cfg = load_cfg("model")
    train_cfg = load_cfg("train")
    
    model = build_model(model_cfg).to(device).eval()
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    actual_save_dir = None if args.no_save else args.save_dir
    
    # 2. Tìm mẫu
    samples = get_clinical_samples(args.data_dir)
    thresholds = train_cfg["composite_score"]["thresholds"]
    
    # 3. Chạy review cho từng ca tìm được
    for task_name, file_path in samples.items():
        if file_path is None:
            print(f"[Warn] Không tìm thấy ca tiêu biểu cho: {task_name.upper()}")
            continue
            
        print(f"\n[Reviewing] Task mục tiêu: {task_name.upper()} | File: {os.path.basename(file_path)}")
        dataset_for_one = ISLES24Dataset([file_path], transform=None)
        sample = dataset_for_one[0]
        inp = sample["input"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(inp)
            
        overlay_predictions(
            sample=sample,
            preds=preds,
            epoch=999, # Mã đánh dấu hậu kiểm
            save_dir=actual_save_dir,
            thresholds=thresholds,
            show=True
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/kaggle/working/outputs/fold_0/checkpoints/best_overall.pt", help="Đường dẫn file .pt")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="clinical_results")
    parser.add_argument("--no_save", action="store_true", help="Chỉ hiển thị ảnh, không lưu file")
    args = parser.parse_args()
    run_clinical_review(args)

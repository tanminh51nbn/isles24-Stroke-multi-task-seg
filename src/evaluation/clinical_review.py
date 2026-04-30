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

def get_clinical_samples(data_dir, num_checks=50):
    """Tìm 3 file npy thỏa mãn điều kiện có LVO, Lesion, CoW"""
    all_files = glob(os.path.join(data_dir, "*.npy"))
    np.random.shuffle(all_files)
    
    found = {"lvo": None, "lesion": None, "cow": None}
    
    print(f"[Review] Đang quét {num_checks} mẫu để tìm ca tiêu biểu...")
    for f in all_files[:num_checks]:
        data = np.load(f, allow_pickle=True).item()
        lbl = data["label"] # (3, H, W)
        
        if found["lvo"] is None and lbl[1].sum() > 5: # Có LVO
            found["lvo"] = f
        if found["lesion"] is None and lbl[0].sum() > 100: # Có Lesion đủ lớn
            found["lesion"] = f
        if found["cow"] is None and lbl[2].sum() > 100: # Có CoW
            found["cow"] = f
            
        if all(v is not None for v in found.values()):
            break
            
    return found


def run_clinical_review(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Config & Model
    with open("src/configs/model.yaml", "r") as f: model_cfg = yaml.safe_load(f)
    with open("src/configs/train.yaml", "r") as f: train_cfg = yaml.safe_load(f)
    
    model = build_model(model_cfg).to(device).eval()
    checkpoint = torch.load(args.model_path, map_location=device)
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
        data = np.load(file_path, allow_pickle=True).item()
        inp = torch.from_numpy(data["input"]).unsqueeze(0).to(device)
        gt = data["label"] # (3, H, W)
        cta = data["input"][6] # CTA trung tâm
        
        with torch.no_grad():
            preds = model(inp)
            
        overlay_predictions(
            sample={"input": torch.from_numpy(data["input"]), "label": torch.from_numpy(data["label"]), "path": file_path},
            preds=preds,
            epoch=999, # Mã đánh dấu hậu kiểm
            save_dir=actual_save_dir,
            thresholds=thresholds,
            show=True
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="clinical_results")
    parser.add_argument("--no_save", action="store_true", help="Chỉ hiển thị ảnh, không lưu file")
    args = parser.parse_args()
    run_clinical_review(args)

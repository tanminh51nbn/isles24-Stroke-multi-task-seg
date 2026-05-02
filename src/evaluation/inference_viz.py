"""
inference_viz.py — Script chạy Inference và Visualization đa năng.

Hỗ trợ 2 chế độ:
    1. --mode single: Chạy 1 mô hình trên N ảnh.
    2. --mode compare: Chạy 3 mô hình (overall, lesion, lvo) trên cùng 1 ảnh để so sánh.
"""

import os
import sys
import torch
import argparse
import yaml
import random
import matplotlib.pyplot as plt

# Tự động thêm thư mục 'src' vào hệ thống để import module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from glob import glob
from torch.utils.data import DataLoader

from models.dual_unet import build_model
from data.dataset import ISLES24Dataset
from evaluation.visualize import overlay_predictions

def load_model_from_ckpt(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model

def run_single_mode(args, model, train_cfg, device):
    # Lấy danh sách file .npy
    all_files = glob(os.path.join(args.data_dir, "*.npy"))
    if not all_files: return print("!!! Không tìm thấy file .npy")
    
    selected_files = all_files[:args.num_samples]
    dataset = ISLES24Dataset(selected_files, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.save_dir, exist_ok=True)
    threshold = train_cfg["composite_score"]["thresholds"].get("lesion", 0.5)

    print(f"[Inference] Đang vẽ {len(selected_files)} mẫu với mô hình {os.path.basename(args.model_path)}...")
    with torch.no_grad():
        for batch in loader:
            inp, path = batch["input"].to(device), batch["path"][0]
            preds = model(inp)
            overlay_predictions(
                sample={"input": inp[0], "label": batch["label"][0], "path": path},
                preds={k: v for k, v in preds.items()},
                epoch=999, save_dir=args.save_dir, 
                thresholds=train_cfg["composite_score"]["thresholds"],
                show=True
            )
    print(f"[Xong] Lưu tại: {args.save_dir}")

def run_compare_mode(args, model, train_cfg, device):
    # 1. Chọn file
    all_files = glob(os.path.join(args.data_dir, "*.npy"))
    if not all_files: return print("!!! Không tìm thấy dữ liệu")
    
    random_file = random.choice(all_files)
    dataset = ISLES24Dataset([random_file], transform=None)
    sample = dataset[0]
    inp = sample["input"].unsqueeze(0).to(device)
    
    # 2. Chạy 3 mô hình và vẽ grid
    names = ["overall", "lesion", "lvo"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Cột 1: Ảnh gốc
    axes[0].imshow(sample["input"][6], cmap='bone') # CTA trung tâm (Vibrant)
    axes[0].set_title(f"Original\n{os.path.basename(random_file)}")
    
    for i, name in enumerate(names):
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_{name}.pt")
        if os.path.exists(ckpt_path):
            model = load_model_from_ckpt(model, ckpt_path, device)
            with torch.no_grad():
                preds = model(inp)
                # Lấy kết quả Lesion làm chuẩn so sánh, dùng ngưỡng từ config
                thresh = train_cfg["composite_score"]["thresholds"].get("lesion", 0.5)
                mask = (torch.sigmoid(preds["lesion"]) > thresh).cpu().numpy()[0, 0]
                axes[i+1].imshow(sample["input"][6], cmap='bone')
                axes[i+1].imshow(mask, cmap='jet', alpha=0.4)
                axes[i+1].set_title(f"Model: {name.upper()}")
        else:
            axes[i+1].text(0.5, 0.5, f"Missing {name}.pt", ha='center')
    
    save_path = os.path.join(args.save_dir, f"compare_{os.path.basename(random_file).replace('.npy','.png')}")
    os.makedirs(args.save_dir, exist_ok=True)
    plt.tight_layout()
    
    if not args.no_save:
        plt.savefig(save_path)
        print(f"[Compare] Đã lưu bảng so sánh tại: {save_path}")
    
    # Hiển thị ảnh trên log (Hoạt động với %run)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISLES'24 Inference Tool")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "compare"])
    parser.add_argument("--model_path", type=str, help="Path tới 1 file .pt (dùng cho single mode)")
    parser.add_argument("--checkpoint_dir", type=str, help="Thư mục chứa 3 file best_*.pt (dùng cho compare mode)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="inference_results")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--no_save", action="store_true", help="Chỉ hiển thị ảnh, không lưu file")
    
    args = parser.parse_args()
    
    # Logic nạp config linh hoạt (Hỗ trợ Kaggle/Local)
    def load_cfg(name):
        paths = [f"src/configs/{name}.yaml", f"configs/{name}.yaml", f"../configs/{name}.yaml"]
        for p in paths:
            if os.path.exists(p):
                with open(p, "r") as f: return yaml.safe_load(f)
        raise FileNotFoundError(f"Không tìm thấy file cấu hình {name}.yaml tại bất kỳ vị trí nào: {paths}")

    model_cfg = load_cfg("model")
    train_cfg = load_cfg("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device).eval()

    if args.mode == "single":
        if not args.model_path: raise ValueError("--model_path là bắt buộc trong single mode")
        model = load_model_from_ckpt(model, args.model_path, device)
        run_single_mode(args, model, train_cfg, device)
    else:
        if not args.checkpoint_dir: raise ValueError("--checkpoint_dir là bắt buộc trong compare mode")
        run_compare_mode(args, model, train_cfg, device)

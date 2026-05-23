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
import numpy as np
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
    
    # Helper chuẩn hóa nhanh
    def _norm(img):
        arr = img.cpu().numpy() if torch.is_tensor(img) else img
        p99 = np.percentile(arr, 99.9)
        arr = np.clip(arr, 0, p99)
        amin, amax = arr.min(), arr.max()
        if amax > amin: arr = (arr - amin) / (amax - amin)
        return (arr * 255).astype(np.uint8)

    # Cột 1: Ảnh gốc + Ground Truth
    bg_img = _norm(sample["input"][0:6].mean(dim=0)) # CTA Anatomy Map
    axes[0].imshow(bg_img, cmap='bone')
    
    # Đè Ground Truth lên (nếu có)
    gt = sample["label"].cpu().numpy()
    if gt.max() > 0:
        ov = np.zeros((*gt[0].shape, 4))
        ov[..., 0] = gt[0] * 0.7  # Lesion: Đỏ
        ov[..., 1] = gt[2] * 0.4  # CoW: Xanh lá
        ov[..., 2] = (gt[1] > 0.1).astype(float) * 0.9 # LVO: Xanh dương
        ov[..., 3] = (gt.max(axis=0) > 0.1) * 0.5
        axes[0].imshow(ov)
        
    axes[0].set_title(f"Original (w/ GT)\n{os.path.basename(random_file)}")
    
    for i, name in enumerate(names):
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_{name}.pt")
        if os.path.exists(ckpt_path):
            model = load_model_from_ckpt(model, ckpt_path, device)
            with torch.no_grad():
                preds = model(inp)
                axes[i+1].imshow(bg_img, cmap='bone')
                
                if name == "lvo":
                    # Hiển thị LVO dạng Heatmap rực rỡ (colormap hot)
                    sig_lvo_tensor = torch.sigmoid(preds["lvo"])
                    lvo_cls = preds.get("lvo_cls", None)
                    if lvo_cls is not None:
                        cls_prob = torch.sigmoid(lvo_cls).view(-1, 1, 1, 1)
                        sig_lvo_tensor = sig_lvo_tensor * cls_prob
                    heat = sig_lvo_tensor.cpu().numpy()[0, 0]
                    axes[i+1].imshow(heat, cmap='hot', alpha=0.6, vmin=0, vmax=1)
                elif name == "lesion":
                    # Lesion hiển thị Mask nhị phân
                    thresh = train_cfg["composite_score"]["thresholds"].get("lesion", 0.5)
                    mask = (torch.sigmoid(preds["lesion"]) > thresh).cpu().numpy()[0, 0]
                    
                    if mask.sum() > 0:
                        ov = np.zeros((*mask.shape, 4))
                        ov[..., 0] = 1.0; ov[..., 3] = mask * 0.5
                        axes[i+1].imshow(ov)
                elif name == "overall":
                    # Overall hiển thị kết hợp cả 3 task (Lesion: Đỏ, LVO: Hot Heatmap, CoW: Xanh lá)
                    thresh_l = train_cfg["composite_score"]["thresholds"].get("lesion", 0.5)
                    thresh_c = train_cfg["composite_score"]["thresholds"].get("cow", 0.5)
                    
                    m_lesion = (torch.sigmoid(preds["lesion"]) > thresh_l).cpu().numpy()[0, 0]
                    m_cow    = (torch.sigmoid(preds["cow"]) > thresh_c).cpu().numpy()[0, 0]
                    sig_lvo_tensor = torch.sigmoid(preds["lvo"])
                    lvo_cls = preds.get("lvo_cls", None)
                    if lvo_cls is not None:
                        cls_prob = torch.sigmoid(lvo_cls).view(-1, 1, 1, 1)
                        sig_lvo_tensor = sig_lvo_tensor * cls_prob
                    h_lvo    = sig_lvo_tensor.cpu().numpy()[0, 0]
                    
                    if m_lesion.sum() > 0:
                        ov_l = np.zeros((*m_lesion.shape, 4))
                        ov_l[..., 0] = 1.0; ov_l[..., 3] = m_lesion * 0.5
                        axes[i+1].imshow(ov_l)
                        
                    if m_cow.sum() > 0:
                        ov_c = np.zeros((*m_cow.shape, 4))
                        ov_c[..., 1] = 1.0; ov_c[..., 3] = m_cow * 0.4
                        axes[i+1].imshow(ov_c)
                        
                    if h_lvo.max() > 0:
                        axes[i+1].imshow(h_lvo, cmap='hot', alpha=0.6, vmin=0, vmax=1)
                    
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

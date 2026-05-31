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

from models import build_model
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

def run_compare_mode(args, base_model, train_cfg, device):
    all_files = glob(os.path.join(args.data_dir, "*.npy"))
    if not all_files: return print("!!! Không tìm thấy dữ liệu")
    np.random.shuffle(all_files)
    
    cases = {}
    print("[Compare] Đang quét tìm 8 trường hợp (dựa trên sự xuất hiện của Lesion, LVO, CoW)...")
    for f in all_files:
        if len(cases) == 8: break
        try:
            ds = ISLES24Dataset([f], transform=None)
            lbl = ds[0]["label"]
        except: continue
        
        has_lesion = bool(lbl[0].sum() > 300)
        has_lvo    = bool(lbl[1].sum() > 3)
        has_cow    = bool(lbl[2].sum() > 200)
        key = (has_lesion, has_lvo, has_cow)
        if key not in cases:
            cases[key] = ds[0]
            print(f"  + Tìm thấy: Lesion={has_lesion}, LVO={has_lvo}, CoW={has_cow}")
    
    keys_order = [
        (False, False, False), (False, False, True),
        (False, True, False), (False, True, True),
        (True, False, False), (True, False, True),
        (True, True, False), (True, True, True)
    ]
    
    # Pre-compute predictions
    names = ["overall", "lesion", "lvo"]
    all_preds = {name: {} for name in names}
    
    for name in names:
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_{name}.pt")
        if os.path.exists(ckpt_path):
            print(f"Đang chạy Inference mô hình: {name}")
            model = load_model_from_ckpt(base_model, ckpt_path, device)
            with torch.no_grad():
                for key in keys_order:
                    if key in cases:
                        inp = cases[key]["input"].unsqueeze(0).to(device)
                        all_preds[name][key] = {k: v.cpu() for k, v in model(inp).items() if torch.is_tensor(v)}
                        
    # Plotting 8x4
    fig, axes = plt.subplots(8, 4, figsize=(20, 40))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    def _norm(img):
        arr = img.cpu().numpy() if torch.is_tensor(img) else img
        p99 = np.percentile(arr, 99.9)
        arr = np.clip(arr, 0, p99)
        amin, amax = arr.min(), arr.max()
        if amax > amin: arr = (arr - amin) / (amax - amin)
        return (arr * 255).astype(np.uint8)

    for row, key in enumerate(keys_order):
        if key not in cases:
            for col in range(4):
                axes[row, col].text(0.5, 0.5, f"Missing case:\nLesion={key[0]}\nLVO={key[1]}\nCoW={key[2]}", ha='center', va='center')
                axes[row, col].axis("off")
            continue
            
        sample = cases[key]
        bg_img = _norm(sample["input"][0:6].mean(dim=0))
        gt = sample["label"].cpu().numpy()
        
        # Col 0: Original + GT
        axes[row, 0].imshow(bg_img, cmap='bone')
        if gt.max() > 0:
            ov = np.zeros((*gt[0].shape, 4))
            ov[..., 0] = gt[0] * 0.7
            ov[..., 1] = gt[2] * 0.4
            ov[..., 2] = (gt[1] > 0.1).astype(float) * 0.9
            ov[..., 3] = (gt.max(axis=0) > 0.1) * 0.5
            axes[row, 0].imshow(ov)
        
        label_str = f"Les={key[0]}, LVO={key[1]}, CoW={key[2]}"
        axes[row, 0].set_title(f"Original (w/ GT)\n{label_str}")
        axes[row, 0].axis("off")
        
        # Col 1, 2, 3: Models
        for i, name in enumerate(names):
            ax = axes[row, i+1]
            ax.axis("off")
            
            if key not in all_preds[name]:
                ax.text(0.5, 0.5, f"Missing {name}.pt", ha='center', va='center')
                continue
                
            preds = all_preds[name][key]
            ax.imshow(bg_img, cmap='bone')
            
            if name == "lvo":
                sig = torch.sigmoid(preds["lvo"])
                ax.imshow(sig.numpy()[0,0], cmap='hot', alpha=0.6, vmin=0, vmax=1)
            elif name == "lesion":
                thresh = train_cfg["composite_score"]["thresholds"].get("lesion", 0.5)
                sig = torch.sigmoid(preds["lesion"])
                mask = (sig > thresh).numpy()[0,0]
                if mask.sum() > 0:
                    ov = np.zeros((*mask.shape, 4))
                    ov[..., 0] = 1.0; ov[..., 3] = mask * 0.5
                    ax.imshow(ov)
            elif name == "overall":
                thresh_l = train_cfg["composite_score"]["thresholds"].get("lesion", 0.5)
                thresh_c = train_cfg["composite_score"]["thresholds"].get("cow", 0.5)
                
                sig_l = torch.sigmoid(preds["lesion"])
                m_l = (sig_l > thresh_l).numpy()[0,0]
                
                m_c = (torch.sigmoid(preds["cow"]) > thresh_c).numpy()[0,0]
                
                sig_v = torch.sigmoid(preds["lvo"])
                h_v = sig_v.numpy()[0,0]
                
                if m_l.sum() > 0:
                    ov_l = np.zeros((*m_l.shape, 4))
                    ov_l[..., 0] = 1.0; ov_l[..., 3] = m_l * 0.5
                    ax.imshow(ov_l)
                if m_c.sum() > 0:
                    ov_c = np.zeros((*m_c.shape, 4))
                    ov_c[..., 1] = 1.0; ov_c[..., 3] = m_c * 0.4
                    ax.imshow(ov_c)
                if h_v.max() > 0:
                    ax.imshow(h_v, cmap='hot', alpha=0.6, vmin=0, vmax=1)
                    
            if row == 0:
                ax.set_title(f"Model: {name.upper()}")

    save_path = os.path.join(args.save_dir, f"compare_8_cases.png")
    os.makedirs(args.save_dir, exist_ok=True)
    if not args.no_save:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[Compare] Đã lưu bảng so sánh 8 trường hợp tại: {save_path}")
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

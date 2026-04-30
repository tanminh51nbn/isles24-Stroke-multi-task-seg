import numpy as np
import os
import pandas as pd
from glob import glob
from tqdm.auto import tqdm

# ─── PATHS ───
DATASET_DIR   = "/kaggle/input/datasets/muynhmuynh/isles24-stroke-segmentation-dataset"
OUTPUT_DIR    = "/kaggle/working/outputs"
METADATA_PATH = "/kaggle/working/dataset_metadata.csv"

CTA_WEIGHTS   = "/kaggle/input/datasets/muynhmuynh/radimagenet-pytorch/ResNet50.pt"
PERF_WEIGHTS  = "/kaggle/input/datasets/muynhmuynh/radimagenet-pytorch/DenseNet121.pt"
# ─── Hàm quét Metadata (Chạy trực tiếp trong Notebook để tránh lỗi import) ───
def scan_dataset_notebook(input_dir, output_csv):
    all_files = glob(os.path.join(input_dir, "*.npy"))
    if not all_files:
        print(f"!!! KHÔNG TÌM THẤY FILE .NPY NÀO TẠI {input_dir} !!!")
        return

    records = []
    for f in tqdm(all_files, desc="Quét Metadata"):
        try:
            # Nạp file với allow_pickle=True
            data = np.load(f, allow_pickle=True)
            
            # TRƯỜNG HỢP 1: data là một Dictionary (thường gặp khi dùng script đóng gói sẵn)
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
            
            if isinstance(data, dict):
                # Giả sử key là 'label' hoặc 'mask'
                label = data.get('label', data.get('mask', None))
                if label is None:
                    continue
            else:
                # TRƯỜNG HỢP 2: data là một Tensor (Cấu trúc cũ)
                if data.shape[0] < 21:
                    label = data[-3:] # Giả định 3 kênh cuối là label
                else:
                    label = data[18:] # Giả định label bắt đầu từ kênh 18
            
            records.append({
                "path": f,
                "has_lesion": int(np.any(label[0] > 0)),
                "has_lvo":    int(np.any(label[1] > 0)),
                "has_cow":    int(np.any(label[2] > 0))
            })
            del data
            
        except Exception as e:
            # In lỗi file đầu tiên để chẩn đoán
            if len(records) == 0:
                print(f"Lỗi chẩn đoán file {os.path.basename(f)}: {e}")
                print(f"Type: {type(data) if 'data' in locals() else 'N/A'}")
            continue
    
    if not records:
        print("!!! KHÔNG CÓ DỮ LIỆU HỢP LỆ ĐƯỢC QUÉT. KIỂM TRA LẠI CẤU TRÚC FILE .NPY !!!")
        return
        
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n--- Hoàn thành! Metadata lưu tại: {output_csv} ---")
    print(df[["has_lesion", "has_lvo", "has_cow"]].sum())

# Thực thi
os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.exists(METADATA_PATH):
    scan_dataset_notebook(DATASET_DIR, METADATA_PATH)
else:
    print(f"Metadata đã tồn tại tại: {METADATA_PATH}")
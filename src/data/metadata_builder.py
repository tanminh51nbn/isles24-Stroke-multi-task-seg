"""
metadata_builder.py — Quét dataset và tạo file CSV thông tin nhãn.
Dùng để hỗ trợ Smart Sampling (Oversampling LVO).
Thiết kế: Có thể chạy độc lập hoặc gọi từ Notebook.
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def scan_dataset(dataset_dir: str, output_csv: str):
    """
    Quét thư mục dataset và trích xuất thông tin nhãn vào file CSV.
    
    Args:
        dataset_dir: Đường dẫn đến thư mục chứa các file .npy
        output_csv:  Đường dẫn lưu file CSV kết quả (nên để ở /kaggle/working)
    """
    print(f"Bắt đầu quét dataset tại: {dataset_dir}")
    file_list = sorted(glob.glob(os.path.join(dataset_dir, "*.npy")))
    
    if not file_list:
        print(f"LỖI: Không tìm thấy file .npy nào trong {dataset_dir}")
        return

    records = []
    for f in tqdm(file_list, desc="Scanning slices"):
        try:
            # Load label (shape 3, 256, 256)
            data = np.load(f, allow_pickle=True).item()
            label = data["label"]
            
            # has_lesion: channel 0, has_lvo: channel 1, has_cow: channel 2
            records.append({
                "filepath": os.path.basename(f),
                "has_lesion": int(np.any(label[0] > 0)),
                "has_lvo": int(np.any(label[1] > 0)),
                "has_cow": int(np.any(label[2] > 0))
            })
        except Exception as e:
            print(f"Lỗi khi xử lý file {f}: {e}")
            continue
        
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Hoàn thành! Metadata được lưu tại: {output_csv}")
    print("\nThống kê số lượng Slice:")
    print(df.sum(numeric_only=True))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quét dataset và tạo metadata cho ISLES'24.")
    parser.add_argument("--input", type=str, default="/kaggle/input/isles24-npy-dataset/ISLES24_NPY_Dataset", help="Đường dẫn thư mục npy")
    parser.add_argument("--output", type=str, default="/kaggle/working/dataset_metadata.csv", help="Đường dẫn lưu file CSV")
    
    args = parser.parse_args()
    scan_dataset(args.input, args.output)

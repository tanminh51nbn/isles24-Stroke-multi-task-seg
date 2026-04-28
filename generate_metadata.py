import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

NPY_DIR  = "/content/drive/MyDrive/Dataset/Stroke/ISLES24_NPY_Dataset"
OUT_CSV  = "/content/drive/MyDrive/Dataset/Stroke/dataset_metadata.csv"

def analyze_slice(fname):
    fpath = os.path.join(NPY_DIR, fname)
    try:
        data = np.load(fpath, allow_pickle=True).item()
        lbl = data['label'] # Shape (3, 256, 256)
        
        # Kiểm tra xem từng kênh nhãn có dữ liệu không
        has_lesion = 1 if lbl[0].max() > 0 else 0
        has_lvo    = 1 if lbl[1].max() > 0 else 0
        has_cow    = 1 if lbl[2].max() > 0 else 0
        
        # Tách Patient ID và Slice Index từ tên file
        # Ví dụ: sub-stroke0001_slice030.npy
        parts = fname.replace(".npy", "").split("_")
        patient_id = parts[0]
        slice_idx  = int(parts[1].replace("slice", ""))
        
        return {
            'filename': fname,
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'has_lesion': has_lesion,
            'has_lvo': has_lvo,
            'has_cow': has_cow
        }
    except:
        return None

def main():
    all_files = [f for f in os.listdir(NPY_DIR) if f.endswith(".npy")]
    print(f"📊 Đang lập hồ sơ cho {len(all_files)} file sạch...")
    
    metadata = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(analyze_slice, all_files), total=len(all_files)))
    
    metadata = [r for r in results if r is not None]
    
    # Tạo DataFrame và lưu CSV
    df = pd.DataFrame(metadata)
    df = df.sort_values(['patient_id', 'slice_idx']).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    
    # In báo cáo tổng kết
    print(f"\n✨ ĐÃ LẬP HỒ SƠ XONG!")
    print(f"📝 File lưu tại: {OUT_CSV}")
    print("-" * 30)
    print(f"📁 Tổng số lát cắt: {len(df)}")
    print(f"👤 Tổng số bệnh nhân: {df['patient_id'].nunique()}")
    print(f"🔴 Số lát có Lesion: {df['has_lesion'].sum()} ({df['has_lesion'].mean()*100:.1f}%)")
    print(f"🟢 Số lát có LVO:    {df['has_lvo'].sum()} ({df['has_lvo'].mean()*100:.1f}%)")
    print(f"🔵 Số lát có CoW:    {df['has_cow'].sum()} ({df['has_cow'].mean()*100:.1f}%)")

if __name__ == "__main__":
    main()

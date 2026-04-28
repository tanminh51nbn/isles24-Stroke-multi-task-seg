import os
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

NPY_DIR = "/content/drive/MyDrive/Dataset/Stroke/ISLES24_NPY_Dataset"

def deep_audit_one_file(fname):
    fpath = os.path.join(NPY_DIR, fname)
    try:
        data = np.load(fpath, allow_pickle=True).item()
        inp = data['input']
        lbl = data['label']
        
        errors = []
        
        # 1. Kiểm tra Shape
        if inp.shape != (18, 256, 256):
            errors.append(f"Sai Input Shape: {inp.shape}")
        if lbl.shape != (3, 256, 256):
            errors.append(f"Sai Label Shape: {lbl.shape}")
            
        # 2. Kiểm tra Range [0, 1]
        if inp.min() < -0.0001 or inp.max() > 1.0001:
            errors.append(f"Sai Range Input: [{inp.min():.4f}, {inp.max():.4f}]")
            
        # 3. Kiểm tra nhãn Binary
        unique_lbl = np.unique(lbl)
        if not np.all(np.isin(unique_lbl, [0, 1])):
            errors.append(f"Nhãn không phải Binary: {unique_lbl}")
            
        if errors:
            return f"❌ {fname}: " + " | ".join(errors)
        return None
    except Exception as e:
        return f"💥 {fname}: Lỗi nạp file - {e}"

def run_deep_audit():
    all_files = [f for f in os.listdir(NPY_DIR) if f.endswith(".npy")]
    print(f"🕵️ Đang thực hiện TỔNG KIỂM SOÁT {len(all_files)} file...")
    
    issues = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(deep_audit_one_file, all_files), total=len(all_files)))
    
    issues = [r for r in results if r is not None]
    
    print("\n" + "="*40)
    print("📋 KẾT QUẢ TỔNG KIỂM SOÁT:")
    if not issues:
        print("✅ TẤT CẢ FILE ĐỀU VƯỢT QUA KIỂM TRA (PASS)!")
        print("🚀 Dữ liệu đã sẵn sàng để Upload lên Kaggle.")
    else:
        print(f"⚠️ Phát hiện {len(issues)} file có vấn đề:")
        for issue in issues[:20]: # In ra 20 lỗi đầu tiên
            print(issue)
        if len(issues) > 20:
            print(f"... và {len(issues)-20} lỗi khác.")
    print("="*40)

if __name__ == "__main__":
    run_deep_audit()

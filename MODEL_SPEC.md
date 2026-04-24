# Model Specification — ISLES24 2.5D Multi-Task Segmentation

Tài liệu này lưu trữ toàn bộ các quyết định về Kiến trúc, Kỹ thuật tối ưu, Hàm Loss, và Workflow Training được áp dụng trong project.

## 1. Kiến trúc Tổng thể (Overall Architecture)
- **Framework Core**: PyTorch + MONAI + segmentation-models-pytorch (SMP)
- **Dạng bài toán**: 2.5D Multi-Task Image Segmentation.
- **Paradigm**: Shared Encoder-Decoder (1 Bộ Encoder & Decoder xài chung, 3 Task Heads độc lập).

## 2. Đầu vào và Đầu ra
### 2.1. Đầu vào (Input)
- **Kích thước Tensor**: `[Batch, 18, 544, 544]`
- **18 Channels bao gồm**: 6 modalities (NCCT, CTA, Tmax, CBF, CBV, MTT) × 3 lát cắt liên tiếp (Z-1, Z, Z+1). Bộ não AI nhìn được độ sâu không gian trên-dưới chứ không chỉ một mặt phẳng.

### 2.2. Đầu ra (Output)
- 3 Tensor độc lập đại diện cho 3 nhãn khác nhau. Mỗi tensor có shape: `[Batch, 1, 544, 544]`.
- Đầu ra là **Raw Logits** (được xử lý Brain Mask với giá trị `-1e9` ở vùng background trước khi tính Loss).

---

## 3. Kiến trúc Chi tiết
### 3.1. Shared Encoder (Bộ trích xuất đặc trưng xài chung)
- **Backbone:** ResNet-50
- **Pretrained:** **RadImageNet** (1.35 triệu ảnh y tế CT/MRI).
- **Kỹ thuật Inflate:** Lớp `conv1` được mở rộng từ 3 lên 18 kênh bằng phương pháp **Average-Repeat**, giúp giữ nguyên dải phân bổ đặc trưng (activation scale) của trọng số pretrained.

### 3.2. Shared Decoder (Bộ giải mã xài chung)
- Sử dụng **Shared UNet Decoder** từ thư viện SMP.
- Việc dùng chung Decoder giúp mô hình học được mối tương quan không gian giữa cục máu đông (LVO) và vùng nhu mô bị tổn thương tương ứng (Lesion).

### 3.3. Multi-Task Heads
- 3 nhánh Conv 1x1 độc lập tạo ra dự đoán cho Lesion, LVO và CoW.

---

## 4. Hàm Loss và Optimizer
### 4.1. Loss Function theo Task
- **Lesion:** `TverskyLoss` (Alpha=0.4, Beta=0.6) — Ưu tiên Recall để không bỏ sót vùng nhồi máu.
- **LVO:** `FocalTverskyLoss` (Gamma=2.0) — Ép model tập trung vào đốm LVO nhỏ li ti và cực hiếm.
- **CoW:** `DiceFocalLoss` — Phân đoạn cấu trúc mạch máu ổn định.

### 4.2. Khâu cân trọng số đa nhiệm (Task Weighting)
- `Loss_Total = 1.0 * L_Lesion + 3.0 * L_LVO + 0.8 * L_CoW`
- LVO được nhân hệ số 3.0 do độ khó và tầm quan trọng lâm sàng cao nhất.

---

## 5. Chiến lược Training (ResNet-Centric)
- **Warmup (Epoch 1-5):** Freeze toàn bộ Encoder. Chỉ cho phép Decoder và Heads học.
- **Joint Training (Epoch 6-50):** Mở toàn bộ mạng.
- **Differential Learning Rate:** 
  - `Encoder LR = 3e-5` (Nhỏ để bảo tồn RadImageNet).
  - `Head/Decoder LR = 3e-4` (Lớn để học task mới).
- **Automatic Mixed Precision (AMP):** Chạy `fp16` để tiết kiệm VRAM và tăng tốc trên Kaggle T4.

---

## 6. Monitoring & Visualization
### 6.1. Metrics
- **Composite Score:** `0.4 * Dice_Lesion + 0.4 * Recall_LVO + 0.2 * Dice_CoW`.
- **Grad Norm:** Theo dõi độ ổn định của gradient để phát hiện outlier.

### 6.2. Visualization (Định kỳ 5 epoch)
- Trainer tự động chọn 4 slice chứa LVO/Lesion để vẽ overlay:
  - Red: LVO | Green: Lesion | Blue: CoW.

---

## 7. Thực nghiệm Lần 0 (Exp 0 — Sanity Check)
Đây là module chạy thử nghiệm đầu tiên để xác nhận tính toàn vẹn của Pipeline trước khi "đốt" GPU cho 50 epoch chính thức.

### 7.1. Cấu hình Exp 0
- **Epochs:** 1 - 2.
- **Batch Size:** 2 (Tối thiểu).
- **Mục tiêu kỹ thuật:**
  - Xác nhận dữ liệu load đúng (Input shape `18, 544, 544`).
  - Xác nhận Loss không bị `NaN`.
  - Xác nhận Brain Mask hoạt động (Ảnh visualization không bị nhiễu ngoài sọ).
  - Xác nhận Gradient Flow (Grad Norm có giá trị, không bằng 0).

### 7.2. Dấu hiệu vượt qua Sanity Check
- Có file `.pth` xuất hiện trong thư mục `checkpoints/`.
- Có file `.png` xuất hiện trong `visualizations/` với hình hài não bộ rõ nét.
- Log không có thông báo lỗi về Device mismatch hoặc Incompatible shapes.

---
_Tài liệu được cập nhật ngày 24/04/2026 bởi Antigravity AI Agent._

# Model Specification — ISLES24 2.5D Multi-Task Segmentation

Tài liệu này lưu trữ toàn bộ các quyết định về Kiến trúc, Kỹ thuật tối ưu, Hàm Loss, và Workflow Training được áp dụng trong project.

## 1. Kiến trúc Tổng thể (Overall Architecture)
- **Framework Core**: PyTorch + MONAI + segmentation-models-pytorch (SMP)
- **Dạng bài toán**: 2.5D Multi-Task Image Segmentation.
- **Paradigm**: Hard Parameter Sharing (1 Bộ Encoder xài chung, 3 Bộ Decoder đẻ ra 3 Output khác nhau).

## 2. Đầu vào và Đầu ra
### 2.1. Đầu vào (Input)
- **Kích thước Tensor**: `[Batch, 18, 544, 544]`
- **18 Channels bao gồm**: 6 modalities (NCCT, CTA, Tmax, CBF, CBV, MTT) × 3 lát cắt liên tiếp (Z-1, Z, Z+1). Bộ não AI nhìn được độ sâu không gian trên-dưới chứ không chỉ một mặt phẳng.

### 2.2. Đầu ra (Output)
- 3 Tensor độc lập đại diện cho 3 nhãn khác nhau. Mỗi tensor có shape: `[Batch, 1, 544, 544]`.
- Đầu ra là Raw Logits (chưa qua activation function, hàm Sigmoid sẽ được áp vào lúc tính Loss hoặc tính Dice metrics nhằm đảm bảo tính ổn định số học).

---

## 3. Kiến trúc Chi tiết
### 3.1. Shared Encoder (Bộ trích xuất đặc trưng xài chung)
- **Backbone:** EfficientNet-B2
- **Điều chỉnh cốt lõi:** Lớp Convolution đầu tiên (`conv_stem`) được sửa đổi để tiếp nhận `in_channels=18` (của Pytorch mặc định chỉ kẹp được ảnh 3 kênh). Trọng số kênh mới được khởi tạo bằng Kaiming Normalization, phần còn lại từ từ tốn dùng Pretrained Weight gốc (ImageNet).

### 3.2. Multi-Decoder (3 Nhánh giải mã)
- **Nhánh 1**: Giải mã vùng Nhồi máu (Lesion) 
- **Nhánh 2**: Giải mã vị trí Tắc nghẽn mạch máu lớn (LVO)
- **Nhánh 3**: Giải mã vòng nối Willis (CoW)
- Cả 3 nhánh đều dùng kiến trúc UNet Decoder tiêu chuẩn (Upsampling + Skip Connections từ lớp sâu thẳm của Encoder). Sự chia nhánh ngay từ dưới phễu cổ chai giúp mô hình có không gian giải mã riêng. Những hình hài dị biệt như phân bố ống nối vạch (Vòng Willis), đốm lốm đốm nhỏ tí hon (LVO), hay dị dạng loang lỗ ngẫu mảng (Nhồi Máu) sẽ không dẫm chân Gradient nhau.

---

## 4. Hàm Loss và Optimizer
### 4.1. Loss Function (Dice + Focal Loss)
- Cấu trúc chung cho từng nhánh: `Loss_branch = (1.0 * DiceLoss) + (1.0 * FocalLoss_gamma_2)`
- Mặc dù Dice Loss tốt ở đoạn tối ưu các cục bự nhưng có xu hướng nổ/khủng khiếp khi gặp vùng Background màu đen lớn khổng lồ mà cục LVO nhỏ li ti. Hàm Focal Loss tham gia làm phao gánh Gradient nổ, và bắt model hạ hình phạt các pixel dễ (đen) đi lại tập trung toàn trọng số vào việc phân bổ pixel điểm sáng đốm tụ nhỏ xíu.

### 4.2. Khâu cân trọng số đa nhiệm (Task Weighting)
- `Loss_Total = 1.0 * Loss_Lesion + 2.0 * Loss_LVO + 1.0 * Loss_CoW`
- Vì LVO là thành phần rất quan trọng nhưng lại mất cân bằng Dataset nghiêm trọng. Hàm Mất mát cho LVO được x2 sức phạt để bắt Model chú ý. (Chiến cơ khuyên là có thể đẩy lên `3.0` lúc làm Fine-tuning).

### 4.3. Optimizer & Tối ưu hóa Learning Rate
- **Thuật toán:** AdamW với Weight Decay `1e-4` để tránh Overfitting.
- **Differential Learning Rate (Bảo tồn Tri thức):**
  - Nhánh Decoders mới khởi tạo trắng tinh: Learning rate trần `3e-4`.
  - Bộ Encoder đã có sẵn võ của ImageNet: Learning rate bóp xuống bằng scale 0.1 (`3e-5`) để mô hình không xóa sổ những gì nó từng học tốt ở tiền kì.
- **Scheduler:** `CosineAnnealingLR` hạ dần xuống vực sâu để bám fit, kèm theo `LinearLR` (Linear Warmup) nhẹ nhàng 5 ép ban đầu tránh sốc nhiệt.
- **Gradient Clipping:** Max norm = 1.0 (Ngăn cơn sóng thần Gradient đẩy rác vào Layer).

---

## 5. Chiến lược Phân bổ Phần cứng (DDP - Kaggle 2x T4 GPU)
- **Sẵn sàng phần cứng:** Tối ưu trần để bung lụa cho cấu trúc máy Kaggle. Code được gói bảo bọc vào 1 hàm để Accelerator tạo đa luồng không đẩy tràn Rác VRAM RAM.
- **Automatic Mixed Precision (AMP):** Mô hình tự ép Float32 qua Float16 trong thời gian Model forward, triệt hạ gánh nặng tải dung lượng dữ liệu lớn mà không mất tính chất Loss.
- **DDP (Distributed Data Parallel):** Module `Accelerate` từ chối trò "tổ kiến chúa DataParallel" (Một Gpu bóc dữ liệu cho con gửi cho thằng Gpu hai, GPU 1 chạy phòi râu gộp kết quả trong khi Gpu hai ngồi chờ lâu). `Accelerate DDP` lập liền 2 luồng độc lập, gán 2 bộ Data vào, ép hai GPU xông pha múc 1 lượt tự tính ngược (Backward) tự update. Cuối cùng nhảy hàm All-Reduce của vòng tròn NCCL gom Gradient về đồng đều (Tốc độ tối đa hóa ~1.9x).

---

## 6. Chiến lược Metrics Chuẩn Y tế (Validation & GĐ Test)
### 6.1. Validation Phase (Đánh giá nóng trong lúc Train, trên GPU)
- Được cấu hình để tính toán cực nhanh, vì nó cần phải nhảy ra ngay giữa các Batch Epoch để làm căn cứ lưu File, Dừng sớm...
- **Composite Score (Chiến thuật Lâm sàng)**: Đoạt điểm Release thay vì cân nhắc đồng điệu:
  `Composite_Score = 0.4 * Dice_Lesion + 0.4 * Recall_LVO + 0.2 * Dice_CoW`.
- *Trick:* Nhánh LVO không đo Dice (vẽ viền đốm sáng cục mạch quá tào lao) mà xài Recall. Có nghĩa là mô hình nhắm trúng cái neo LVO thì điểm + ngay!

### 6.2. Test Phase (Đánh giá Nguội sau Training, siêu chuyên sâu bằng CPU)
Được cấu hình bằng hàm `evaluator.py`, sau khi Training xong, mang model ra, load data bệnh nhân, Scan và Stack hàng loạt dự đoán cắt lát lên lại thành khối Khối 3D Volummetric.
- **HD95 (Hausdorff 95th Percentile)**: Dùng lib y tế `MONAI`. Lấy đường kính 95% sai số lớn nhất giữa đường viền mô hình tự vẽ so sánh chênh lệch mm so với nét Bác sỹ, điểm khoảng cách ngàn milimet càng thấp vẽ chênh lệch càng đỉnh.
- **Object-level F1 (LVO)**: Xác nhận sự tinh tế để ứng dụng vào khám lân sàng. Máy thay bộ Pixel thành Point quét lấy Tâm máu đông. Nếu Vùng cảnh báo máy đưa ra liếm đè vào Vùng Tròn (Radius R=3 pixels) xung quanh tâm này sẽ đếm là "1 Lần Phát Hiện Thành Công". Lọc cảnh báo giả rác tốt.

### 6.3 Checkpointing (3 Đầu ra)
1. `best_overall_model.pth` (Dựa trên Composite Score tổng hợp).
2. `best_lesion_model.pth` (Dựa trên Dice Lesion tinh chuẩn).
3. `best_lvo_model.pth` (Dựa trên Recall của LVO, phát hiện báo động siêu nhanh).

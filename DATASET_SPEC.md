# ISLES'24 - 18-Channel 2.5D Stroke Segmentation Dataset

Tài liệu này là đặc tả kỹ thuật chính thức và hướng dẫn sử dụng chi tiết cho bộ dữ liệu ISLES'24 đã qua tiền xử lý, được thiết kế chuyên biệt cho việc huấn luyện các mô hình Deep Learning trong chẩn đoán đột quỵ.

---

## 1. Ngữ cảnh ra đời (Context & Genesis)

Bộ dữ liệu nguyên bản ISLES 2024 (được cung cấp qua Zenodo) là một kho báu y khoa nhưng lại đi kèm với những thách thức kỹ thuật cực kỳ hóc búa. Dữ liệu thô bao gồm ảnh cắt lớp không cản quang (NCCT), ảnh mạch máu (CTA), và các bản đồ tưới máu não (Perfusion map: Tmax, CBF, CBV, MTT).

**Những thách thức từ dữ liệu gốc:**
- **Lệch hệ không gian (Spatial Misalignment):** Các bản chụp có độ phân giải, khung giới hạn (bounding box) và gốc tọa độ Z hoàn toàn khác biệt (ví dụ: gốc Z lệch nhau từ -453mm đến -201mm).
- **Lỗi siêu dữ liệu (Corrupted Headers):** Các file phái sinh (`space-ncct`) thường bị lỗi ma trận không gian (ví dụ: `sform_code=0`), khiến các thư viện xử lý ảnh y tế chuẩn như `nilearn` hay `SimpleITK` không thể chiếu đúng không gian vật lý, dẫn đến sự sai lệch hoàn toàn khi nhân ma trận.
- **Nhiễu xương sọ (Skull Interference):** Xương sọ trên ảnh CTA có độ cản quang (Hounsfield Units - HU) trùng lặp với mạch máu có thuốc cản quang. Nếu không loại bỏ, AI sẽ không thể phân biệt được đâu là xương, đâu là điểm tắc mạch (LVO).
- **Khác biệt số lượng lát cắt (FOV Discrepancy):** Perfusion map chỉ chụp một vùng hẹp ở lõi não (khoảng 16 lát cắt), trong khi NCCT/CTA chụp toàn bộ sọ não (~75 lát cắt).

**Quy trình "Luyện kim" (Từ Thô đến Tinh khiết):**
Để biến mớ hỗn độn trên thành bộ dữ liệu sẵn sàng cho huấn luyện, chúng tôi đã triển khai một (Pipeline) nghiêm ngặt:
1. **Đồng bộ hóa Affine (Affine Synchronization):** Ép buộc toàn bộ các hệ ảnh (CTA, Perfusion map, Labels) phải nhận chung một ma trận tọa độ không gian (Affine) của NCCT và sửa lỗi Header (`sform_code=1`), đảm bảo sự khớp nối 100% theo từng Voxel.
2. **Lột sọ bảo tồn (Conservative Skull-Stripping):** Ứng dụng HD-BET để tạo mặt nạ não, sau đó làm phình (dilate) 3 pixel để đảm bảo không vô tình cắt phạm vào các mạch máu nằm sát vỏ não. Toàn bộ nền (background) được đưa về 0 tuyệt đối.
3. **Ép chuẩn lâm sàng (Clinical Windowing):** Cắt lọc các khoảng HU và giới hạn lâm sàng đặc thù cho từng loại ảnh, sau đó chuẩn hóa Min-Max nghiêm ngặt về đoạn `[0, 1]`.
- CTA_w1: [0, 90]
- CTA_w2: [60, 400]
- Tmax: [0, 7]
- CBF: [0, 35]
- CBV: [0, 10]
- MTT: [0, 20]
4. **Đúc 2.5D & Tích hợp MIP (Hybrid 2.5D & MIP Stacking):** Để tạo bối cảnh không gian rộng mà không tốn chi phí tính toán của 3D-CNN, dữ liệu được xếp chồng theo chiều Z bằng cách kết hợp lát cắt gốc và ảnh chiếu cường độ tối đa (MIP): `[MIP_Below(6), Center_Z(6), MIP_Above(6)]`. Hình ảnh được resize về kích thước chuẩn `256x256` thông qua thuật toán padding kết hợp nội suy để bảo toàn tỷ lệ khung hình (Những ai bé hơn 256 thì pad vào, những ai lớn hơn 256 thì resize về).
5. **Thanh lọc bóng ma (Ghost Purging):** Những lát cắt ở đỉnh đầu hoặc dưới cổ (nơi không chứa bất kỳ nhu mô não nào) được tự động quét và loại bỏ vĩnh viễn, tạo ra một bộ dataset "tinh khiết" không chứa dữ liệu rác.

---

## 2. Thông tin và Hướng dẫn sử dụng (Usage Guide)

Bộ dữ liệu này được lưu trữ dưới định dạng `NumPy (.npy)` - định dạng nạp dữ liệu siêu tốc độ cho PyTorch và TensorFlow.

### 2.1. Cấu trúc thư mục
Tất cả các file nằm chung trong một thư mục duy nhất:
```text
ISLES24_NPY_Dataset/
├── sub-stroke0001_slice000.npy
├── sub-stroke0001_slice001.npy
├── ...
└── sub-stroke0188_slice072.npy
```

### 2.2. Định dạng tên file
Cú pháp: `sub-stroke[ID]_slice[Z].npy`
- **`[ID]`**: Mã bệnh nhân (ví dụ: `0001`). **ĐẶC BIỆT LƯU Ý:** Khi chia tập Train/Validation/Test, phải chia theo nhóm `[ID]` (GroupKFold) để tránh rò rỉ dữ liệu (Data Leakage) giữa các lát cắt của cùng một người.
- **`[Z]`**: Vị trí lát cắt gốc.

### 2.3. Cấu trúc bên trong mỗi file `.npy`
Mỗi file là một `Dictionary` chứa 2 khối dữ liệu (Tensors):

#### 🟩 `input`: Shape `(18, 256, 256)`, Kiểu dữ liệu `float32`
Chứa 6 loại ảnh y tế được sắp xếp lai giữa lát cắt đơn gốc và ảnh MIP (Max Intensity Projection) của các lát lân cận trong bán kính quét $WINDOW\_RADIUS = 7$ lát cắt:
- **Channel 00 -> 05 (MIP Below - Quét từ Z-7 đến Z-1):** `[CTA_w1, CTA_w2, Tmax, CBF, CBV, MTT]` (Ảnh chiếu cường độ tối đa)
- **Channel 06 -> 11 (Center_Z - Lát cắt đơn tại vị trí Z):** `[CTA_w1, CTA_w2, Tmax, CBF, CBV, MTT]` (Giữ nguyên độ sắc nét gốc để phân vùng tổn thương)
- **Channel 12 -> 17 (MIP Above - Quét từ Z+1 đến Z+7):** `[CTA_w1, CTA_w2, Tmax, CBF, CBV, MTT]` (Ảnh chiếu cường độ tối đa)

*(Lưu ý về Dữ liệu khuyết: Do đặc thù FOV hẹp của Perfusion map, ở một số lát cắt trên cao hoặc dưới thấp, các kênh Perfusion map (Tmax, CBF, CBV, MTT) sẽ mang giá trị 0 toàn bộ. CTA sẽ vẫn có hình ảnh. Mô hình của bạn phải có khả năng học cách bỏ qua Perfusion map và tự chẩn đoán bằng CTA ở các lát cắt này).*

#### 🟥 `label`: Shape `(3, 256, 256)`, Kiểu dữ liệu `uint8`
Chứa 3 lớp mặt nạ (Mask) nhị phân (Giá trị 0 hoặc 1), tương ứng với lát cắt trung tâm `(Z)`.
- **Channel 0 (`Lesion`):** Vùng lõi nhồi máu (Infarct core).
- **Channel 1 (`LVO`):** Điểm tắc nghẽn mạch máu lớn (Large Vessel Occlusion).
- **Channel 2 (`CoW`):** Cấu trúc giải phẫu hệ đa giác Willis.

### 2.4. Thống kê nhãn & Phân phối đồng thời (Label Distribution & Co-occurrence)

Dưới đây là bảng thống kê chi tiết sự xuất hiện đồng thời của các nhãn trên từng lát cắt (Slice-level co-occurrence) được tính trên toàn bộ bộ dữ liệu:

| Lesion | LVO | CoW | Số lượng lát cắt ban đầu | Số lượng lát cắt sau khi Upsampling
| :---: | :---: | :---: | :---: | :---: |
| ✗ | ✗ | ✗ | 4552 | 1365
| ✗ | ✗ | ✓ | 672 | 672
| ✗ | ✓ | ✗ | 26 | 468
| ✗ | ✓ | ✓ | 46 | 414
| ✓ | ✗ | ✗ | 3215 | 1,607 (Cyclic Stride)
| ✓ | ✗ | ✓ | 711 | 711  
| ✓ | ✓ | ✗ | 14 | 420
| ✓ | ✓ | ✓ | 74 | 444
|| **Tổng** || **9441** | **5701 (Max=7309)**

> [!NOTE]
> Nhóm **Lesion only** (`lesion=✓ lvo=✗ cow=✗`) chiếm tới 27.5% tổng số lát cắt. Nếu giữ nguyên 100%, chúng sẽ chiếm đa số và lấn át các nhãn hiếm như LVO trong batch. Do đó, việc giảm 50% lượng Lesion-only (Cyclic Stride 50%) là bắt buộc để cân bằng tỷ lệ phân bổ lớp.

---


## 3. Mục đích sử dụng (Intended Use Cases)

Bộ dataset này là một bài test (benchmark) cực kỳ mạnh mẽ cho các kiến trúc Deep Learning tiên tiến, mở ra tiềm năng cho các bài toán:

1. **Multi-task Segmentation (Phân vùng đa nhiệm):** Huấn luyện một mạng Neural duy nhất có khả năng chia 3 nhánh (3-head decoder) để đồng thời: (a) Vẽ vùng đột quỵ, (b) Chỉ điểm cục máu đông, (c) Lập bản đồ mạch máu.
2. **Nghiên cứu kiến trúc 2.5D/3D-Aware:** Hoàn hảo để thử nghiệm các kiến trúc như `DenseNet-UNet`, `ResUNet`, hoặc `SegFormer`, nơi các lớp tích chập (Convolution) có thể khai thác sự liền mạch theo chiều Z thông qua 18 kênh Channel đầu vào.
3. **Mô hình chống chịu khuyết thiếu Modality (Modality-Missing Robustness):** Giải quyết một bài toán lâm sàng rất phổ biến: Chẩn đoán đột quỵ chính xác ngay cả khi bệnh viện không có máy chụp Perfusion map, hoặc dữ liệu Perfusion map bị lỗi/nhiễu, bằng cách ép mô hình tập trung suy luận từ CTA.

# Báo Cáo EDA — ISLES24 2.5D NPY Dataset
**Ngày phân tích:** 22/04/2026  
**Dataset:** ISLES'24 — A Real-World Longitudinal Multimodal Stroke Dataset (v7)  
**Mục tiêu:** Phân tích đặc điểm dữ liệu để đưa ra quyết định thiết kế mô hình segmentation đa nhãn

---

## 1. Tổng Quan Dataset

| Thuộc tính | Giá trị |
|---|---|
| Tổng số bệnh nhân (case) | **149** |
| Tổng số slice | **11,821** |
| Số slice trung bình / case | **79.3** |
| Số slice tối thiểu / case | ~31 |
| Số slice tối đa / case | ~140 |
| Kích thước input | `[18, 544, 544]` — float32 |
| Kích thước label | `[3, 544, 544]` — uint8 binary |

**Nhận xét:** Phần lớn bệnh nhân có từ 75–90 slice (chiếm ~90% dataset), tương ứng vùng từ basal ganglia lên đỉnh não sau resample 1mm/slice. Một số ít case outlier có ~31–40 slice do perfusion kéo Z-min xuống thấp (đã ghi nhận trong spec).

---

## 2. Thống Kê Nhãn (Label Statistics)

### 2.1 Tỷ lệ case và slice dương tính

| Nhãn | Cases có nhãn | Slices dương tính | Imbalance ratio |
|---|---|---|---|
| **Lesion** | 140 / 149 (94.0%) | 4,956 / 11,821 (41.9%) | 1 : 213 |
| **LVO** | 139 / 149 (93.3%) | 379 / 11,821 (**3.2%**) | **1 : 213,282** |
| **CoW** | 147 / 149 (98.7%) | 3,285 / 11,821 (27.8%) | 1 : 584 |

**⚠️ Phát hiện nghiêm trọng — LVO:**  
Tỷ lệ mất cân bằng của LVO là **1:213,282** — đây là mức extreme imbalance. Chỉ có 16,401 pixel dương tính trên tổng 3.5 tỷ pixel toàn dataset. Đây là thách thức lớn nhất của toàn bộ bài toán.

### 2.2 Phân bố kích thước nhãn per slice

| Nhãn | Min (px) | Median (px) | Mean (px) | Max (px) |
|---|---|---|---|---|
| Lesion | 1 | 1,399 | 3,306 | 32,521 |
| LVO blob | 1 | **25** | 43.3 | 355 |
| CoW blob | 1 | **69** | ~100 | 700+ |

**Nhận xét Lesion:**  
Long-tail distribution nặng — mean (3,306) gấp đôi median (1,399), cho thấy một số case có vùng nhồi máu rất lớn kéo trung bình lên. Nhiều slice chỉ có 1–5 pixel lesion (viền vùng hoại tử).

**Nhận xét LVO:**  
Median chỉ 25px, max 355px — tính ra trên ảnh 544×544 thì LVO chiếm chưa đến **0.12%** diện tích slice. Đây thực chất là bài toán **detection** hơn là segmentation pixel-perfect.

**Nhận xét CoW:**  
Multi-blob nhỏ (median 69px/blob), tổng 11,252 blobs toàn dataset. Mỗi blob đại diện cho một đoạn động mạch của vòng Willis tại một slice cụ thể.

### 2.3 Số LVO blobs per slice dương tính

| Số blobs | Số slices | Tỷ lệ |
|---|---|---|
| 1 blob | ~365 | ~96% |
| 2 blobs | ~14 | ~4% |

**→ Bilateral LVO (2 blobs) rất hiếm (~4%).** Thiết kế metric `object_f1_centroid` với radius=3 hoàn toàn phù hợp.

---

## 3. Co-occurrence Giữa Các Nhãn

### 3.1 Case-level

| Tổ hợp | Số cases |
|---|---|
| Có cả 3 nhãn (lesion + lvo + cow) | **135 / 149 (90.6%)** |
| Chỉ lesion + cow (không có lvo) | 4 |
| Chỉ lesion (không có lvo, cow) | 1 |
| Chỉ cow (không có lesion, lvo) | 4 |
| Không có nhãn nào | 1 |

**→ 90.6% case có đầy đủ cả 3 nhãn.** Lesion và LVO gần như không tồn tại độc lập — đây là bằng chứng mạnh ủng hộ **multi-task learning với shared encoder**: encoder sẽ học được correlation sinh lý giữa cục máu đông (LVO) và vùng hoại tử (Lesion).

### 3.2 Slice-level

| Tổ hợp | Số slices | Tỷ lệ |
|---|---|---|
| Không có nhãn nào | 5,191 | **43.9%** |
| Chỉ cow | 1,500 | 12.7% |
| Chỉ lesion | 3,249 | 27.5% |
| Lesion + cow | 1,502 | 12.7% |
| Có lvo (mọi tổ hợp) | 379 | 3.2% |
| Cả 3 nhãn | 163 | 1.4% |

**⚠️ 43.9% slice hoàn toàn trống (all-negative).** Nếu train không có chiến lược sampling, model sẽ bị bias mạnh về phía "predict 0 hết" do quá nhiều negative examples.

---

## 4. Phân Bố Không Gian

### 4.1 Vị trí theo trục Z (chiều cao não)

| Nhãn | Median Z | Phân bố |
|---|---|---|
| Lesion | **0.74** | Tập trung phần trên não (Z=0.6–1.0) |
| LVO | **0.55** | Phân bố chuẩn, trung tâm não |
| CoW | **0.56** | Phân bố chuẩn, trung tâm não |

**Phát hiện quan trọng:**  
Ba nhãn có **vùng Z khác nhau rõ rệt**:
- LVO và CoW nằm ở vùng **basal ganglia / giữa não** (Z≈0.4–0.7) — đây là vị trí giải phẫu học của Circle of Willis và các động mạch lớn
- Lesion (vùng nhồi máu) nằm **cao hơn** (Z≈0.6–1.0) — vùng não bị thiếu máu do tắc mạch phía dưới

**→ Ảnh hưởng đến sampling:** Slice ở Z thấp (0.0–0.3) gần như không có lesion nhưng vẫn có thể có CoW và LVO. Không nên loại bỏ hoàn toàn các slice này.

### 4.2 Vị trí theo không gian X-Y

**LVO centroids (n=389):**
- Tập trung vùng **X: 200–350, Y: 150–450** — lệch sang **trái** so với trung tâm ảnh
- Hot-spot rõ ràng tại khoảng (270, 210) và (280, 350) — tương ứng vị trí động mạch não giữa (MCA) trái và phải
- Không đồng đều: LVO bên trái nhiều hơn bên phải (phù hợp với y văn — stroke MCA trái chiếm ưu thế)

**CoW centroids (n=11,252):**
- Phân bố **Gaussian 2D** rất đặc ở trung tâm, radius ~100px từ tâm (272, 300)
- Đây là đặc trưng giải phẫu học ổn định — Circle of Willis luôn nằm ở trung tâm nền sọ

**→ CoW rất predictable về vị trí.** Đây là lợi thế — model có thể học prior vị trí này nhanh chóng.

---

## 5. Phân Tích Cường Độ Pixel Per Modality

### 5.1 Kết quả đo được

| Modality | Mean | Std | Min | Max | Vấn đề |
|---|---|---|---|---|---|
| NCCT | -0.583 | 0.647 | -1.000 | +1.000 | Spike khổng lồ tại -1.0 |
| CTA | -0.642 | 0.524 | -1.000 | +1.000 | Spike khổng lồ tại -1.0 |
| Tmax | 0.005 | 0.402 | -0.726 | **4.438** | Outlier nhẹ |
| CBF | 0.002 | 0.409 | -0.430 | **34.594** | ⚠️ Outlier cực nặng |
| CBV | 0.001 | 0.223 | **-10.773** | **19.500** | ⚠️ Outlier cực nặng |
| MTT | 0.001 | 0.175 | **-46.938** | 5.449 | ⚠️ Outlier cực nặng |

### 5.2 Phân tích vấn đề

**NCCT và CTA — Background padding:**  
~80% pixel có giá trị = -1.0 (minimum của range [-1,+1]). Đây là vùng **ngoài não** (background/air) được clamped về -1 sau crop 544×544. Mô hình cần phân biệt "không có thông tin" vs "có thông tin nhưng giá trị thấp".

**Perfusion (CBF, CBV, MTT) — Outlier nghiêm trọng:**  
- CBF max=34.594 trong khi std=0.409 — outlier này gấp **84 lần std**
- MTT min=-46.938 trong khi std=0.175 — outlier này gấp **268 lần std**  

Nguyên nhân: zscore per-channel non-zero không loại trừ được các giá trị artifact từ vùng ngoài não hoặc lỗi acquisition. Các outlier này sẽ làm gradient không ổn định trong quá trình train.

### 5.3 Khuyến nghị xử lý bổ sung

```python
# Trong DataLoader, sau khi load x:
x = x.astype(np.float32)

# 1. Clip outlier perfusion channels (index 6-17)
x[6:18] = np.clip(x[6:18], -5.0, 5.0)

# 2. Tạo brain mask từ NCCT (channel 1 = slice giữa của NCCT)
brain_mask = (x[1] > -0.95).astype(np.float32)  # shape [544, 544]
# Dùng brain_mask để mask loss: chỉ tính loss trong vùng não thật
```

---

## 6. Tóm Tắt Các Phát Hiện & Quyết Định Thiết Kế

### 6.1 Bảng tóm tắt phát hiện

| # | Phát hiện | Mức độ ảnh hưởng |
|---|---|---|
| 1 | LVO imbalance 1:213,282 — extreme | 🔴 Critical |
| 2 | 43.9% slice all-negative | 🔴 Critical |
| 3 | CBF/CBV/MTT có outlier cực nặng | 🔴 Critical |
| 4 | NCCT/CTA bị dominated bởi background (-1.0) | 🟠 High |
| 5 | Lesion size biến động 1→32,521 px (long-tail) | 🟠 High |
| 6 | LVO và CoW tập trung vùng Z=0.4–0.7 | 🟡 Medium |
| 7 | CoW có vị trí X-Y rất ổn định (Gaussian tại trung tâm) | 🟡 Medium |
| 8 | 90.6% case có cả 3 nhãn → multi-task correlation cao | 🟢 Positive |
| 9 | 149 cases → dataset nhỏ, cần regularization mạnh | 🟠 High |
| 10 | Lesion tập trung Z cao (0.74) khác LVO/CoW (0.55/0.56) | 🟡 Medium |

### 6.2 Quyết định kiến trúc mô hình

**Backbone:** `MiT-B2` (SegFormer)
- Đủ capacity cho 3 task nhưng không quá nặng với 149 cases
- Hierarchical attention giúp CoW (cần global context để "thấy" toàn bộ vòng Willis)
- Pretrained: SegFormer-B2 trên ImageNet-1k, inflate `conv1` từ 3→18 channels bằng phương pháp average-repeat

**Decoder:** Shared lightweight MLP decoder (theo SegFormer design) + 3 task heads riêng biệt

**Loss function per task:**

| Task | Loss | Lý do |
|---|---|---|
| Lesion | `TverskyLoss(α=0.4, β=0.6)` | Long-tail, ưu tiên recall |
| LVO | `FocalTverskyLoss(α=0.2, β=0.8, γ=2.0)` | Extreme imbalance, penalize FN nặng |
| CoW | `DiceFocalLoss` | Multi-blob nhỏ, cân bằng precision-recall |

**Loss weights tổng hợp:**
```python
total_loss = 1.0 * loss_lesion + 3.0 * loss_lvo + 0.8 * loss_cow
```

**Sampling strategy:**
- Slice có bất kỳ nhãn nào: lấy 100%
- Slice all-negative: chỉ lấy **30%** ngẫu nhiên mỗi epoch
- Slice có LVO: **oversampling 3×** (do quá hiếm — 3.2% tổng slices)

**Xử lý input bổ sung (trong DataLoader):**
- Clip perfusion channels [6:18] về `[-5, 5]`
- Tạo brain_mask từ NCCT để áp dụng vào loss computation

**Metric đánh giá:**

| Task | Metric chính | Lý do |
|---|---|---|
| Lesion | Dice Score | Standard segmentation metric |
| LVO | Object-level F1 (centroid radius=3) | Blob quá nhỏ, pixel Dice không ý nghĩa |
| CoW | Dice Score | Multi-blob binary segmentation |

---

## 7. Rủi Ro Và Hạn Chế

1. **Overfitting cao** do chỉ 149 cases. Cần data augmentation mạnh (flip, rotate, elastic deformation, intensity jitter) và dropout trong decoder.

2. **LVO có thể không học được** nếu loss weight hoặc sampling không được điều chỉnh đúng. Cần monitor LVO F1 riêng biệt và sẵn sàng tách thành model riêng nếu cần.

3. **1 case không có nhãn nào** — cần kiểm tra lại case này để xác định có phải lỗi annotation không hay bệnh nhân thực sự không có tổn thương.

4. **5 case có Z ngắn (~31–43 slices)** do vấn đề perfusion FOV — các case này sẽ thiếu thông tin vùng não dưới, cần xử lý riêng trong DataLoader (padding hoặc flag).

5. **Outlier perfusion cực nặng** (MTT min=-46.938) có thể là artifact từ acquisition — nếu clip [-5,5] không đủ, cần xem xét per-patient normalization.

---

*Báo cáo này được tổng hợp từ kết quả EDA script chạy trên toàn bộ 149 cases / 11,821 slices của ISLES24 dataset.*

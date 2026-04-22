# Dataset Specification — ISLES24 2.5D NPY

## Tổng quát dataset gốc

Dataset gốc: ISLES'24 - A Real-World Longitudinal Multimodal Stroke Dataset | Version 7

### Mô tả

- Dataset này chụp từ vùng basal ganglia lên đến đỉnh não, không phải toàn bộ đầu. Độ dài vùng này từ 70-90mm (70-90 lát cắt dày 1mm).
- Số lương mẫu: 149 bệnh nhân (ID từ 1 đến 189, không đánh số liên tiếp).
- Định dạng .nii.gz
  > Có 5 bệnh nhân bất thường nhưng hợp lệ vì bị Perfusion kéo z min xuống: `Z gốc = 8 slices × spacing 5mm = 40mm FOV → sau resample 1mm ra ~36 slices`.

| Case | Z gốc | Spacing Z | FOV thực | Sau resample 1mm |
| ---- | ----- | --------- | -------- | ---------------- |
| 0004 | 69    | 2.0mm     | 138mm    | 36               |
| 0012 | 75    | 2.0mm     | 150mm    | 36               |
| 0030 | 64    | 2.0mm     | 128mm    | 36               |
| 0037 | 48    | 3.0mm     | 144mm    | 43               |
| 0066 | 68    | 2.0mm     | 136mm    | 31               |

### Phân tích cấu trúc

- raw_data: là dữ liệu thô chưa qua xử lý, lấy ngay sau khi bệnh nhân được chụp CT. Gồm: perfusion (cbf, cbfv, mtt, tmax), ncct, cta, ctp.
  [ *Chỉ cần ncct* ]
- derivatives: là dữ liệu đã qua xử lý, mọi dữ liệu trong này đã được xử lí về không gian ncct (space_ncct). Gồm hai session:
  - ses-01:
    Input: gồm perfusion (cbf, cbfv, mtt, tmax), cta, ctp.
    Mask: lvo-msk, cow-msk.
    [ *Không dùng ctp* ]
  - ses-02: chỉ gồm lesion-msk, dwi, adc.
    [ *Không dùng dwi, adc* ]
- Phenotype: dữ liệu dạng bảng (.csv) chứa thông tin lâm sàn.

**Tóm lại:** Bộ dataset này đầy đủ các đặc trưng, không thiếu.

- INPUT: Sử dụng `ncct` trong raw_data để làm khung, ép các kênh thô khác khớp với nó vì nó chứa đầy đủ sơ đồ hình não chi tiết. `CTA` (Ảnh chụp CT Angiography , đây là sơ đồ máu) và `perfusion` gồm: `CBF` (Lưu lượng máu), `CBV` (Thể tích máu), `MTT` (Thời gian trung bình máu đi qua não), `Tmax` (Thời gian máu đạt đỉnh).

- LABEL: `lesion-msk` (vùng nhồi máu - hoại tử), `lvo-msk` (vị trí cục máu đông gây tắc mạch), `cow-msk` (phục vụ phân loại vòng nối gồm các động mạch lớn).
  >- lesion: Vùng lớn, hình dạng bất quy tắc => Class imbalance nặng (vùng tổn thương nhỏ so với toàn não)
  >- lvo: Điểm trên mạch máu, cực nhỏ => dễ bị model lướt qua.
  >- cow: Cấu trúc mạng mạch máu dạng vòng => Mỏng, dài, topology phức tạp.

## Tái cấu trúc thư mục và Tiền xử lý

Chuyển từ .nii.gz sang .npy nhị phân thô để tăng tốc độ train, đồng thời tiền xử lí và chuẩn hóa sẵn luôn.

```
ISLES24_NPY_544x544x1mm/
└── sub-XXXX/
    ├── inputs/
    │   ├── x_z000.npy
    │   ├── x_z001.npy
    │   └── ...
    └── labels/
        ├── y_z000.npy
        ├── y_z001.npy
        └── ...
```

Bộ dữ liệu là hình các hình chụp không hoàn hảo, bệnh nhân không ở yên một và tùy cách xử lí hình, chúng ta sẽ tìm kích thước phù hợp dựa trên vùng tọa độ chứa nhãn cực biên (Global Bounding Box):

- Trục X: [42 tới 511] -> Độ rộng cần: 469
- Trục Y: [50 tới 575] -> Độ cao cần: 525

  => Vậy ta có `Size` lớn nhất là `469 × 525` px. Chọn Resize thành `544 × 544` px.

- Về chiều dày, chúng ta có thống kê như sau:

| Chiều dày lát cắt (mm) | Số lượng bệnh nhân | Tỷ lệ (%) |
| ---------------------- | ------------------ | --------- |
| 0.4000                 | 67                 | 45.0      |
| 0.7500                 | 44                 | 29.5      |
| 0.4500                 | 8                  | 5.4       |
| 0.6000                 | 7                  | 4.7       |
| 0.9000                 | 3                  | 2.0       |
| 2.0000                 | 2                  | 1.3       |
| 3.0000                 | 2                  | 1.3       |
| 1.5000                 | 2                  | 1.3       |
| 1.0500                 | 1                  | 0.7       |
| 0.3900                 | 1                  | 0.7       |
| 0.9540                 | 1                  | 0.7       |
| 0.9720                 | 1                  | 0.7       |
| 0.9700                 | 1                  | 0.7       |
| 0.9870                 | 1                  | 0.7       |
| 1.0170                 | 1                  | 0.7       |
| 0.4100                 | 1                  | 0.7       |
| 1.0000                 | 1                  | 0.7       |
| 2.0002                 | 1                  | 0.7       |
| 1.6000                 | 1                  | 0.7       |
| 2.5000                 | 1                  | 0.7       |
| 0.9490                 | 1                  | 0.7       |
| 0.8000                 | 1                  | 0.7       |

=> Vậy độ dày phù hợp để bộ dataset này không bị nén và giãn quá mức là 1mm cho một lát cắt

`Vậy chúng ta sẽ chọn kích thước là 544x544x1mm`

## Input (`x_zXXX.npy`)

### Thông tin chung

| Thuộc tính   | Giá trị                                    |
| ------------ | ------------------------------------------ |
| Shape        | `[18, 544, 544]`                           |
| Dtype        | `float16`                                  |
| Channels     | 18 = 6 modalities × 3 slices (z-1, z, z+1) |
| Edge padding | Clamp — lặp lại slice biên                 |

### Thứ tự channels

| Index | Modality | Phương pháp normalize             | Range      |
| ----- | -------- | --------------------------------- | ---------- |
| 0–2   | NCCT     | ScaleIntensityRange `[0, 80]`     | `[-1, +1]` |
| 3–5   | CTA      | ScaleIntensityRange `[-150, 300]` | `[-1, +1]` |
| 6–8   | Tmax     | zscore nonzero per-channel        | `~N(0,1)`  |
| 9–11  | CBF      | zscore nonzero per-channel        | `~N(0,1)`  |
| 12–14 | CBV      | zscore nonzero per-channel        | `~N(0,1)`  |
| 15–17 | MTT      | zscore nonzero per-channel        | `~N(0,1)`  |

## Label (`y_zXXX.npy`)

### Thông tin chung

| Thuộc tính | Giá trị         |
| ---------- | --------------- |
| Shape      | `[3, 544, 544]` |
| Dtype      | `uint8`         |
| Giá trị    | Binary `{0, 1}` |

### Thứ tự channels

| Index | Channel  |
| ----- | -------- |
| 0     | `lesion` |
| 1     | `lvo`    |
| 2     | `cow`    |

## Spatial

| Thuộc tính         | Giá trị                              |
| ------------------ | ------------------------------------ |
| Resolution XY      | 544×544 px (center crop sau padding) |
| Slice thickness Z  | 1.0 mm (resampled)                   |
| Interpolation ảnh  | Bilinear                             |
| Interpolation nhãn | Nearest                              |

**KẾT LUẬN**

Toàn bộ dataset đều nằm gọn trong trong khung crop `544x544` với cách cắt theo trọng tâm của ảnh, hướng phát triển là chọn trọng tâm của não thay vì cả ảnh. Dưới đây là 4 case cực biên:

```
sub-stroke0096:
  Shape: (512, 577, 74), Center: (256, 288)
  Crop X: [-16, 528]      Não X: [85, 511] | ✓
  Crop Y: [16, 560]       Não Y: [161, 526] | ✓

sub-stroke0117:
  Shape: (512, 512, 69), Center: (256, 256)
  Crop X: [-16, 528]      Não X: [81, 353] | ✓
  Crop Y: [-16, 528]      Não Y: [50, 345] | ✓

sub-stroke0153:
  Shape: (512, 560, 44), Center: (256, 280)
  Crop X: [-16, 528]      Não X: [42, 286] | ✓
  Crop Y: [8, 552]        Não Y: [104, 483] | ✓

sub-stroke0182:
  Shape: (512, 667, 41), Center: (256, 333)
  Crop X: [-16, 528]      Não X: [179, 443] | ✓
  Crop Y: [61, 605]       Não Y: [210, 575] | ✓
```

## Lưu ý khi train

> **Bắt buộc** cast input về `float32` trong dataloader trước khi đưa vào model.

```python
x = np.load(path).astype(np.float32)
```

- NCCT và CTA ở `[-1, +1]`, Perfusion ở `~N(0,1)` — **không normalize thêm**
- Slice index `z` trong tên file tương ứng với slice trung tâm của 2.5D stack
- Mỗi file input chứa context 3 slices liên tiếp, không phải 1 slice đơn lẻ
- Dataset được đẩy lên Kaggle (136.43 GB), nó được chia làm 2 phần:
  - Part 1 (64.17 GB): gồm 75 case đầu tiên (isles24-stroke-dataset-part-1)
  - Part 2 (72.26 GB): gồm 74 case cuối cùng (isles24-stroke-dataset-part-2)

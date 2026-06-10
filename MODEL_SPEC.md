# ISLES'24: Technical Whitepaper - Single-Encoder Multi-Task 2.5D UNet (SOTA Architecture)

Tài liệu này đặc tả toàn bộ kiến trúc mô hình, hệ thống hàm mất mát (Loss) và chiến lược huấn luyện đa nhiệm (Multi-Task Learning) thực tế đang được áp dụng trong codebase dành cho bài toán phân vùng đột quỵ (ISLES 2024).

---

## 1. Tổng Quan Kiến Trúc (Architecture Overview)

Kiến trúc chính là **Single-Encoder Triple-Decoder UNet** với cơ chế rẽ nhánh sâu tuần tự (**Knowledge Cascade**) và tích hợp tưới máu trực tiếp (**Raw Perfusion Injection**). Mô hình thực hiện Early Fusion 18 kênh �```mermaid
graph TD
    subgraph InputProcessing ["Input & Preprocessing"]
        Input["Raw Input: 18 Channels (2.5D)<br/>[6 CTA + 12 CTP]<br/>Shape: (B, 18, 256, 256)"]
        SA["SliceAttention<br/>(Channel Reweighting)"]
        Input --> SA
    end

    subgraph EncoderBlock ["Encoder (DenseNet-121)"]
        Conv0_A["Stream A (CoW/LVO)<br/>Conv 3x3, Stride 2<br/>Shape: (B, 32, 128, 128)"]
        Conv0_B["Stream B (Lesion)<br/>Conv 9x9, Stride 2<br/>Shape: (B, 32, 128, 128)"]
        Concat_Conv0["Concat conv0<br/>Shape: (B, 64, 128, 128)"]
        
        Enc1["Encoder Block 1<br/>Shape: (B, 256, 64, 64)"]
        Enc2["Encoder Block 2<br/>Shape: (B, 512, 32, 32)"]
        Enc3["Encoder Block 3<br/>Shape: (B, 1024, 16, 16)"]
        Bot["DenseGlobalBottleneck<br/>(7x7 DW, Expand 4x)<br/>Shape: (B, 512, 16, 16)"]
        
        SA --> Conv0_A
        SA --> Conv0_B
        Conv0_A --> Concat_Conv0
        Conv0_B --> Concat_Conv0
        Concat_Conv0 --> Enc1 --> Enc2 --> Enc3 --> Bot
    end

    subgraph SharedDecoder ["Shared Decoder (Low-Res)"]
        Dec4["dec4 (Shared)<br/>Up + DualAttention<br/>Shape: (B, 256, 32, 32)"]
        Dec3["dec3 (Shared)<br/>Up + DualAttention<br/>Shape: (B, 128, 64, 64)"]
        Bot --> Dec4
        Enc2 -.->|"Skip"| Dec4
        Dec4 --> Dec3
        Enc1 -.->|"Skip"| Dec3
    end

    subgraph TaskDecoders ["Task-Specific Decoders (High-Res)"]
        FiLM["Task-Conditioned FiLM<br/>(Tô màu đặc trưng)"]
        Dec3 --> FiLM
        Concat_Conv0 -.->|"Skip"| FiLM

        %% CoW Branch
        Dec2_CoW["CoW dec2<br/>Shape: (B, 64, 128, 128)"]
        Dec1_CoW["CoW dec1<br/>Shape: (B, 32, 256, 256)"]
        
        %% LVO Branch
        Asym_LVO["Hemispheric Asymmetry<br/>(High-Pass Filter)"]
        Dec2_LVO["LVO dec2<br/>Shape: (B, 64, 128, 128)"]
        Dec1_LVO["LVO dec1<br/>Shape: (B, 32, 256, 256)"]
        
        %% Lesion Branch
        Dec2_Les["Lesion dec2<br/>Shape: (B, 64, 128, 128)"]
        Dec1_Les["Lesion dec1<br/>Shape: (B, 32, 256, 256)"]

        FiLM --> Dec2_CoW
        FiLM --> Dec2_LVO
        FiLM --> Dec2_Les

        Dec2_CoW --> Dec1_CoW
        
        Dec2_LVO --> Asym_LVO --> Dec1_LVO
        
        %% Guidance
        Dec1_CoW -.->|"CoW.detach()<br/>FusedSpatialAttn"| Dec1_LVO

        %% Perfusion Physics Encoder
        PhysEnc["PerfusionPhysicsEncoder<br/>(10 Channels Input:<br/>6 raw + core + penumbra + mismatch + tmax_asym)"]
        Input -.-> PhysEnc
        PhysEnc -.->|"perf_feat_64<br/>(64ch, 64x64)"| Dec2_Les
        PhysEnc -.->|"perf_feat_128<br/>(32ch, 128x128)"| Dec1_Les
    end

    subgraph Heads ["Segmentation Heads"]
        Head_CoW["CoW Head<br/>(SE + ResBlock)<br/>Bias: -2.944"]
        Head_LVO["LVO Head<br/>(SE + ResBlock)<br/>Bias: -3.000"]
        Head_Les["Lesion Head<br/>(SE + ResBlock)<br/>Bias: -2.944"]

        Dec1_CoW --> Head_CoW
        Dec1_LVO --> Head_LVO
        Dec1_Les -->|"Explicit Perf Mask"| Head_Les
    end

    subgraph Outputs ["Final Output"]
        Out_CoW["CoW Mask<br/>Shape: (B, 1, 256, 256)"]
        Out_LVO["LVO Heatmap<br/>Shape: (B, 1, 256, 256)"]
        Out_Les["Lesion Mask<br/>Shape: (B, 1, 256, 256)"]
        
        Head_CoW --> Out_CoW
        Head_LVO --> Out_LVO
        Head_Les --> Out_Les
    end
end
```

### 1.1. Backbone (Encoder) & Dual-Stream Input Processing
*   **SliceAttention:** Khối Attention dạng Squeeze-and-Excitation cải tiến nằm ngay trước `conv0` nhằm gán trọng số ưu tiên cho các lát cắt.
*   **Dual-Stream Shallow Feature Extractor:** Lớp `conv0` của DenseNet-121 được phá bỏ và thay bằng 2 luồng hoạt động song song ngay từ đầu:
    *   **Stream A (3x3, 32 channels):** Tối ưu hóa để bắt các dải viền siêu sắc nét, mạch máu nhỏ (CoW, LVO).
    *   **Stream B (9x9, 32 channels):** Receptive Field cực lớn để thu nhận các dải biến thiên chậm, nhu mô não, vùng mờ (Lesion).
*   **Backbone:** Các tầng sâu (denseblocks) tải trọng số y học **RadImageNet** (99.9% tham số được giữ nguyên để bảo tồn transfer learning).

### 1.2. Shared & Specialized Decoders với Task-Conditioned FiLM
*   **Shared Bottleneck & Path:** Cát nghĩa không gian chung ở độ phân giải thấp ($16\times16$ và $32\times32$) bằng khối **DenseGlobalBottleneck**.
*   **Task-Conditioned FiLM:** Tại "ngã ba" rẽ nhánh, các skip connections và shared features phải đi qua bộ lọc FiLM. Mỗi Task (CoW, LVO, Lesion) sử dụng 1 vector embedding riêng để ép FiLM biến đổi (Scale & Shift) dải đặc trưng cho phù hợp với khẩu vị của mình, giải phóng Encoder khỏi xung đột tín hiệu.
*   **Task-Specific Decoder Paths:** Phân tách thành 3 đường giải mã độc lập tại ($64\times64$, $128\times128$).
*   **Hemispheric Asymmetry Modules:** Cung cấp tư duy bác sĩ điện quang (so sánh não trái-não phải):
    *   **LVO Asymmetry (High-Pass Filter):** Trừ mảng mờ, giữ lại các đốm nhọn và sáng (cục LVO).

### 1.3. Cơ Chế Kết Nối Nâng Cao
*   **Knowledge Cascade (Lan truyền tri thức):** 
    Nhánh **LVO** nhận đặc trưng dẫn đường từ CoW (`CoW.detach()`) qua khối **FusedSpatialAttention** để thu hẹp vùng tìm kiếm điểm tắc mạch dọc theo cấu trúc cây mạch máu. *(Nhánh Lesion đã được CẮT ĐỨT khỏi CoW để tránh ô nhiễm bởi tần số cao của mạch máu mảnh).*
*   **PerfusionPhysicsEncoder & Perfusion Injection:** 
    Nhánh **Lesion** tích hợp tri thức lâm sàng qua `PerfusionPhysicsEncoder` nâng đầu vào thô lên 10 kênh đặc trưng vật lý y học (bao gồm 6 kênh perfusion gốc, core_map, penumbra_map, mismatch_map và tmax_asym). Encoder phụ này trích xuất hai dải đặc trưng đa quy mô: `perf_feat_128` (32 kênh, kích thước $128\times128$) và `perf_feat_64` (64 kênh, kích thước $64\times64$), được nối trực tiếp vào các skip connections của Lesion decoder (`dec1` và `dec2`). Cùng với đó, `perf_mask` (0 hoặc 1) được bơm thẳng vào Lesion Head để mạng tự nhận biết tình trạng thiếu hụt dữ liệu CTP và chuyển hướng phân tích.

### 1.4. Task-Specific Segmentation Heads
Mỗi nhánh ra mặt nạ được thiết kế dưới dạng: `ChannelAttention(reduction=4) -> ResidualBlock(2x Conv3x3) -> Dropout2d -> Conv1x1`.
*   **Bias Initialization Đặc Thù:** Khởi tạo bias để giải quyết mất cân bằng lớp cực đoan và dập tắt báo ảo (FP) trên nền âm tính:
    *   **LVO Head:** `bias = -3.0` (tương đương baseline probability $\sigma \approx 0.047$). Ngưỡng này ép mô hình phải "chắc chắn" mới dám vượt ngưỡng 0.4.
    *   **CoW & Lesion Head:** `bias = -2.944`�ng tự nhận biết tình trạng thiếu hụt dữ liệu CTP và chuyển hướng phân tích.

### 1.4. Task-Specific Segmentation Heads
Mỗi nhánh ra mặt nạ được thiết kế dưới dạng: `ChannelAttention(reduction=4) -> ResidualBlock(2x Conv3x3) -> Dropout2d -> Conv1x1`.
*   **Bias Initialization Đặc Thù:** Khởi tạo bias để giải quyết mất cân bằng lớp cực đoan và dập tắt báo ảo (FP) trên nền âm tính:
    *   **LVO Head:** `bias = -3.0` (tương đương baseline probability $\sigma \approx 0.047$). Ngưỡng này ép mô hình phải "chắc chắn" mới dám vượt ngưỡng 0.4.
    *   **CoW & Lesion Head:** `bias = -2.944`

---

## 2. Hệ Thống Hàm Mất Mát (Task-Specific Losses)

Do tính chất hình học và phân phối nhãn rất khác biệt, mỗi nhánh sử dụng một tổ hợp Loss chuyên biệt phối hợp cùng Deep Supervision.

| Nhiệm vụ | Công thức Loss | Rationale (Lý do thiết kế) |
| :--- | :--- | :--- |
| **Lesion** (Ổ nhồi máu) | `Focal Tversky (batch=True, α=0.35, β=0.65, γ=1.75)` <br>+ `Soft Lesion Labeling` | **Batch-level Tversky:** Gom toàn bộ pixel trong batch để tính chung 1 giá trị TI, tránh pha loãng gradient bởi các lát cắt rỗng. <br>**Soft Labeling:** Phân tách và làm mềm nhãn thật (về dải 0.1 - 0.95) theo các vùng Core/Penumbra/Benign, giúp mô hình bớt bảo thủ và bắt được các vùng rìa tranh tối tranh sáng chưa hoại tử. |
| **LVO** (Tắc mạch lớn) | `Modified Focal Loss (α=2.5, β=4.5)` <br>+ `Gaussian Curriculum` <br>+ `Negative Slice Max Penalty` | LVO chỉ là dạng một chấm nhỏ. **Gaussian Curriculum** tạo vùng Gaussian quanh điểm tắc, thu nhỏ dần $\sigma$ (xuống mức floor=1.25) theo Epoch để dễ hội tụ.<br>**Negative Slice Max Penalty:** Phạt bình phương giá trị dự đoán tối đa ($max\_pred^2$) trên các lát cắt không có LVO để đè bẹp các báo ảo (False Positives). |
| **CoW** (Vòng Willis) | `Tversky Loss (α=0.2, β=0.8)` <br>+ `Soft CLDice Loss (weight=0.45)` | `β=0.8` phạt rất nặng hiện tượng đứt gãy mạch máu. **clDice (Centerline Dice)** sử dụng skeletonization mềm để duy trì tính liên tục và topology dạng mạng lưới của mạch máu. |

---

## 3. Cân Bằng Đa Nhiệm (Multi-Task Balancing)

Để huấn luyện đồng thời 3 tác vụ mà không bị hiện tượng một tác vụ dễ lấn át tác vụ khó, mô hình áp dụng cơ chế đánh trọng số động:

### 3.1. PGW (Performance Gap Weighting)
Trọng số các tác vụ được cập nhật tự động (với momentum) dựa trên khoảng cách tới mục tiêu (Gap):
*   **Mục tiêu hiệu năng:** `Lesion_Dice = 0.75`, `LVO_D2C_F1 = 0.50`, `CoW_Dice = 0.88`.
*   **Cập nhật:** $\text{Gap}_t = \max(0, \text{Target}_t - \text{Metric}_t)$. Task nào càng xa mục tiêu, Loss của task đó càng được khuếch đại ở Epoch tiếp theo.

---

## 4. Chiến Lược Đánh Giá & Hậu Xử Lý Lâm Sàng (Evaluation)

*   **Đánh giá LVO cấp độ Bệnh nhân (Patient-level Majority Voting):** Để giảm triệt để báo ảo (1 lát cắt nhiễu kéo theo cả bệnh nhân bị chẩn đoán nhầm), logic đánh giá quy định một bệnh nhân chỉ được kết luận là dương tính với LVO khi có **ít nhất 2 lát cắt (min_pos_slices = 2)** vượt ngưỡng dự đoán (threshold > 0.4).
*   **Volumetric Dice cho Lesion:** Dice của Lesion được tính trên phương diện không gian 3D (gom toàn bộ pixel của batch) thay vì lấy trung bình các lát cắt, khớp hoàn toàn với phương pháp huấn luyện Batch-level Focal Tversky.

---

## 5. Chiến Lược Tối Ưu Hóa (Optimization Protocol)

*   **Warm-up Curriculum:** Encoder được đóng băng (`frozen`) trong 11 Epoch đầu. Từ Epoch 12, Encoder được mở băng và huấn luyện với tốc độ học nhỏ hơn (Differential LR tỷ lệ 20:1).
*   **3-Phase Scheduler:** Kết hợp SequentialLR bao gồm: Warmup (5 Epoch) $\rightarrow$ Hold (7 Epoch) $\rightarrow$ Cosine Annealing (88 Epoch).
*   **Independent Checkpointing:** Lưu trữ 3 mô hình tốt nhất độc lập (`best_overall.pt`, `best_lesion.pt`, `best_lvo.pt`).
*   **Modality Robustness:** Tích hợp `ModalityDropout` (ngẫu nhiên tắt kênh CTA hoặc CTP xác suất 10%) giúp mô hình có khả năng suy luận ổn định khi thiếu khuyết modality.

---

## 6. Trọng Điểm Của Mô Hình (Những Chi Tiết Nhỏ Sống Còn)

Kiến trúc đa nhiệm y tế thường sụp đổ bởi những mâu thuẫn ngầm. Dưới đây là 5 "trọng điểm" tưởng chừng nhỏ bé nhưng lại là mấu chốt quyết định sự thành bại của toàn bộ hệ thống Single-Encoder này:

1. **Dual-Stream Shallow Features (`conv0_A` và `conv0_B`)**
   - **Vấn đề:** Mạch máu (CoW) cần kernel nhỏ để giữ độ sắc nét. Vùng nhồi máu (Lesion) cần kernel to để thu nhặt dải cường độ mờ. Ép Encoder học chung 1 bộ kernel 7x7 dẫn đến *Catastrophic Interference* ngay từ lớp đầu tiên.
   - **Giải pháp:** Tách `conv0` thành 2 luồng: `5x5` (Stream A) và `9x9` (Stream B). Bộ não điện tử được tự do trích xuất "cạnh sắc" và "mảng mờ" một cách song song mà không dẫm chân lên nhau.

2. **Kỹ Thuật Phân Tách Hemispheric Asymmetry (High-Pass vs Low-Pass)**
   - **Vấn đề:** So sánh 2 bán cầu não là tư duy kinh điển của bác sĩ. Nhưng LVO (cục máu đông) là một hạt đậu sáng chóe, còn Lesion là một bãi bùn đen khổng lồ. 
   - **Mấu chốt:** 
     - LVO Asymmetry sử dụng **High-Pass Filter** (`diff - blur`) để dập tắt các mảng bất đối xứng to đùng do Lesion gây ra, chỉ cho hạt đậu lọt qua.
     - Lesion Asymmetry sử dụng **Low-Pass Filter** (`AvgPool2d`) để xóa sạch các đốm sáng nhiễu li ti, giữ nguyên mảng bùn đen nguyên vẹn.
   - Nếu cắm nhầm 2 module này cho nhau, mô hình sẽ hoàn toàn "mù lòa".

3. **Cắt Đứt Hoàn Toàn CoW Guidance Cho Lesion**
   - **Vấn đề:** Trong khi LVO cần bản đồ mạch máu CoW để biết cục tắc nằm trên đường nào, Lesion hoàn toàn không cần nhìn thấy mạch máu vi mô để xác định mảng chết. Việc ép Lesion nhìn vào bản đồ CoW chỉ gây ô nhiễm phổ tần (High-frequency noise) và triệt tiêu Gradient (do dropout bản đồ).
   - **Mấu chốt:** Tách hoàn toàn Lesion khỏi CoW. Lesion chỉ dùng thông tin Perfusion gốc và Asymmetry Module. Càng cắt bớt thông tin rác, mô hình càng thông minh.

4. **Trấn Áp LVO False Positive Bằng "Top-16 Mean Penalty"**
   - **Vấn đề:** Ở các lát cắt âm tính, phạt điểm cực đại `amax` là quá dễ lách luật. Mạng Neural gian lận bằng cách xả một đám mây báo ảo có xác suất mờ mờ `0.45` trên một vùng rộng. Điểm `amax` chỉ là `0.45`, bình phương lên `0.2` (mức phạt quá rẻ), nhưng lúc test lại bị lộ là False Positive cực lớn.
   - **Mấu chốt:** Thay `amax` bằng trung bình của **Top-16 pixel** (tương đương khối 4x4 pixel). Nếu dám xả một đám mây báo ảo, lập tức cả khối mây này sẽ kích hoạt mức phạt khủng khiếp, diệt tận gốc ý định lách luật của mô hình.

5. **Safe Perfusion Bottleneck & Explicit Mask Injection**
   - **Vấn đề:** Không phải lát cắt nào cũng có ảnh tưới máu (CTP). Nếu đưa một tensor toàn số 0 vào chuẩn hóa `InstanceNorm`, phương sai sẽ bằng 0 và layer bị vỡ.
   - **Mấu chốt:** Viết một Bypass an toàn trong Bottleneck. Thêm vào đó, gắn 1 flag (0 hoặc 1) thẳng vào Lesion Head cuối cùng. "Này Head, lát cắt này không có Perfusion đâu, mi hãy nhìn bằng CTA đi". Mô hình không phải đoán mò tín hiệu bị thiếu nữa, nó chủ động chuyển trạng thái suy luận.
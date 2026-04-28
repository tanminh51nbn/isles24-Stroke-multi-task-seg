# ISLES'24: Technical Whitepaper - Dual-Encoder Multi-Task 2.5D UNet (v4)

## 1. Model Architecture Deep-Dive

### 1.1 Dual-Encoder (Backbones)
To optimally extract both anatomical and functional features, the architecture utilizes two specialized encoders processing distinct sets of input channels from the 18-channel 2.5D stack.

#### CTA Branch (Structural)
- **Base:** ResNet-50.
- **Input:** 6 channels (CTA_w1, CTA_w2 from slices Z-1, Z, Z+1).
- **Purpose:** Extracts sharp, high-resolution anatomical boundaries and identifies hyperdense vessels/clots.
- **Inflation:** `conv1` weight ($W \in \mathbb{R}^{64 \times 3 \times 7 \times 7}$) is transformed to $\mathbb{R}^{64 \times 6 \times 7 \times 7}$ using variance-preserving replication.
- **Pre-trained Weights:** RadImageNet (optimized for CT/MRI).

#### Perfusion Branch (Functional)
- **Base:** DenseNet-121.
- **Input:** 12 channels (Tmax, CBF, CBV, MTT from slices Z-1, Z, Z+1).
- **Purpose:** Extracts semantic, regional changes in blood flow. Dense connections are highly effective at preserving soft gradient information (penumbra).
- **Inflation:** `conv0` weight is transformed from 3 to 12 channels using variance-preserving replication.
- **Pre-trained Weights:** RadImageNet.

### 1.2 Multi-Level Fusion & Shared Decoder
- **Fusion:** Feature maps from both encoders are concatenated at every skip-connection level (resolutions 1/2, 1/4, 1/8, 1/16, 1/32).
- **Bottleneck:** Deepest features (2048 from ResNet + 1024 from DenseNet = 3072 channels) are concatenated and compressed to 1024 channels.
- **Decoder:** 5-stage UNet decoder upsamples and refines the combined structural-functional representations, culminating in a 16-channel feature map.

### 1.3 Multi-Task Segmentation Heads
- **Architecture:** `Conv3x3 -> BN -> ReLU -> SpatialDropout2d(0.3) -> Conv1x1`.
- **Heads:** Three independent heads output raw logits for Lesion, LVO, and CoW.
- **Spatial Dropout:** Applied to entire feature maps (channels) rather than individual pixels to prevent co-adaptation in highly correlated medical imaging data.

## 2. Data Engineering & Augmentation

### 2.1 Patient-Level Split & Sampling
- **GroupShuffleSplit:** Ensures slices from the same patient (`sub-stroke[ID]`) are strictly kept in the same fold to prevent Data Leakage.
- **Sampling Strategy:** 
  - Downsamping negative (background-only) slices to 30%.
  - Oversampling rare LVO-positive slices by a factor of 5.

### 2.2 Augmentation Pipeline
- **Spatial (Synchronized on Input & Label):** Random Horizontal Flip (50%), Random Affine (Rotation $\pm 15^\circ$, Scale $\pm 10\%$, Translate $\pm 10px$). Interpolation uses Bilinear for inputs and Nearest-Neighbor for labels to preserve binary integrity.
- **Intensity (Input Only):** Gaussian Noise ($\mu=0, \sigma=0.05$) and Intensity Scaling ($\pm 10\%$).

## 3. Training & Optimization Logic

### 3.1 Task-Specific Loss Formulation
The total loss is a weighted sum: $\mathcal{L}_{total} = \mathcal{W}_{Lesion}\mathcal{L}_{Lesion} + \mathcal{W}_{LVO}\mathcal{L}_{LVO} + \mathcal{W}_{CoW}\mathcal{L}_{CoW}$
- **$\mathcal{L}_{Lesion}$ (Tversky):** $\alpha=0.4, \beta=0.6, \mathcal{W}=1.0$. Penalizes False Negatives more to ensure stroke core coverage.
- **$\mathcal{L}_{LVO}$ (Focal Tversky):** $\alpha=0.2, \beta=0.8, \gamma=3.0, \mathcal{W}=10.0$. Heavily penalizes missed occlusions and focuses the gradient on hard-to-detect tiny LVOs. Given the highest clinical priority (10x multiplier).
- **$\mathcal{L}_{CoW}$ (Tversky):** $\alpha=0.5, \beta=0.5, \mathcal{W}=0.5$. Balanced segmentation of major vessels.

### 3.2 Optimization Protocol
- **Mixed Precision:** FP16 (AMP) with `GradScaler` for memory efficiency on Kaggle T4 GPUs.
- **Gradient Clipping:** Max norm = 1.0 to prevent gradient explosions driven by the highly weighted LVO loss.
- **Differential Learning Rate:**
  - Base LR: $1.0 \times 10^{-4}$ (Decoder + Heads).
  - Encoder LR: $1.0 \times 10^{-5}$ (Base LR $\times 0.1$). Protects the RadImageNet pre-trained weights from catastrophic forgetting.
- **Scheduler:** 8-epoch Linear Warmup followed by Cosine Annealing.
- **Freeze Strategy:** Encoders are frozen for the first 5 epochs to allow the randomly initialized decoder to stabilize.

## 4. Evaluation & Clinical Checkpointing

### 4.1 Metrics
- **Volumetric Dice:** Evaluates Lesion and CoW spatial overlap.
- **Recall (Sensitivity):** Primary metric for LVO. Clinically, detecting the presence of an occlusion is far more critical than pixel-perfect boundaries.
- **Composite Score:** $0.4 \times \text{Dice}_{Lesion} + 0.4 \times \text{Recall}_{LVO} + 0.2 \times \text{Dice}_{CoW}$.

### 4.2 Tri-Faceted Checkpointing
The pipeline automatically saves three clinical variants of the model:
1. `best_overall.pt`: Highest Composite Score (Primary deployment model).
2. `best_lesion.pt`: Highest Lesion Dice (Optimized for infarct volume estimation).
3. `best_lvo.pt`: Highest LVO Recall (Optimized for emergency triage and clot detection).

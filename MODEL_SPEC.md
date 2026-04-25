# ISLES'24: Technical Whitepaper - Multi-Task 2.5D Stroke Segmentation (v3)

## 1. Model Architecture Deep-Dive

### 1.1 Encoder (Backbone)
- **Base:** ResNet50.
- **Input Manifold:** 18 channels (NCCT x 1, CTA x 1, Perfusion Maps x 16).
- **Inflation Layer:** The first `conv1` weight ($W \in \mathbb{R}^{64 \times 3 \times 7 \times 7}$) is transformed into $W_{new} \in \mathbb{R}^{64 \times 18 \times 7 \times 7}$ using:
  $$W_{new} = \text{Repeat}(\text{Mean}(W, \text{dim}=1), 18) \times \frac{3}{18}$$
  This ensures that the initial feature activation variance remains consistent with the pre-trained RadImageNet scale.
- **Pre-trained Weights:** RadImageNet (specifically curated for radiology features like density shifts and vascular structures).

### 1.2 Shared Decoder & Multi-Head Branching
- **Decoder:** Feature Pyramid style with skip-connections from ResNet blocks.
- **Bottleneck:** Shared representation layer (16 channels) before splitting into task-specific heads.
- **Head Separation:** Each head is a $1 \times 1$ Convolutional layer followed by an implicit Sigmoid (handled during loss for numerical stability).

## 2. Data Engineering & Augmentation

### 2.1 Task-Balanced Sampling (Strict Ratio)
To combat the sparsity of LVO (Large Vessel Occlusion), the sampler draws from 4 independent pools:
- **LVO Pool:** Positive for arterial occlusion.
- **Lesion Pool:** Positive for ischemic core (but LVO negative).
- **CoW Pool:** Positive for Circle of Willis (Anatomy).
- **Negative Pool:** Pure background/healthy tissue.
**Batch Composition (N=24):** 3 LVO, 1 Lesion, 2 CoW, 1 Neg, 17 Mixed.

### 2.2 Spatial & Intensity Augmentation
- **Affine Transforms:** Random Rotation ($\pm 15^\circ$), Scaling ($\pm 10\%$), Translation ($\pm 10$ pixels).
- **Interpolation:** Bilinear for images; **Nearest Neighbor** for labels to preserve boundary integrity.
- **Intensity Shifts:** Random Gaussian Noise ($\sigma=0.05$) and Random Intensity Scaling ($\pm 10\%$).
- **LVO Softening:** Gaussian kernel ($\sigma=5px$) applied to LVO masks to create a learning gradient, while maintaining the "Hard Core" (GT=1.0) using `np.maximum`.

## 3. Training & Optimization Logic

### 3.1 Loss Function Formulation
The total loss is a weighted sum: $\mathcal{L}_{total} = \sum \lambda_i \mathcal{L}_i$
- **$\mathcal{L}_{Lesion}$ (Tversky):** $\alpha=0.4, \beta=0.6$. Penalizes False Negatives more to ensure stroke coverage.
- **$\mathcal{L}_{LVO}$ (Focal Tversky):** $\gamma=3.0$. Focuses the gradient on hard-to-detect small occlusions.
- **$\mathcal{L}_{CoW}$ (Tversky):** $\alpha=0.5, \beta=0.5$. Balanced segmentation of major vessels.

### 3.2 Optimization Protocol
- **Mixed Precision:** FP16 (AMP) for 2x faster throughput on T4 GPUs.
- **Learning Rate:** $1.0 \times 10^{-4}$ with **Differential Scaling**.
  - Encoder LR: $1.0 \times 10^{-5}$ (Fine-tuning).
  - Decoder/Heads LR: $1.0 \times 10^{-4}$ (Active learning).
- **Warmup:** 8-epoch linear ramp-up to prevent gradient explosion in early iterations.
- **Freezing:** 5-epoch encoder freeze to allow the shared decoder to stabilize.

## 4. Evaluation & Inference Workflow

### 4.1 Distributed 3D Inference
- **Memory Optimization:** 3D volumes are processed in mini-batches of 16 slices.
- **Aggregation:** Slices are reconstructed into $[Z, H, W]$ volumes.
- **Post-processing:** Skull-masking is applied to zero out non-brain predictions.

### 4.2 Metrics
- **Volumetric Dice:** Measures spatial overlap for Lesion and CoW.
- **Object-level F1:** Centroid-based matching with a $3mm$ (3 voxel) tolerance. This is the primary clinical metric for LVO detection.
- **Zero-Bias Aggregation:** Uses Global Sum/Count across all DDP ranks to ensure mathematical precision of the final report.

---
*Target Hardware: Kaggle Dual T4 (2x16GB VRAM)*
*Execution Time: ~11 Hours for 35 Epochs*

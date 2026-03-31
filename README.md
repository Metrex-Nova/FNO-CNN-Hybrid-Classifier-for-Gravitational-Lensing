Rewritten README with corrected architecture, training logic, and results:

---

# Neural Operator Classifier

Gravitational lensing image classifier using a **Fourier Neural Operator (FNO)** combined with an **EfficientNet-B0** backbone.

---

## Architecture

A dual-stream model where spectral and spatial representations are learned in parallel and fused for classification.

**FNO branch**
Processes the input in the Fourier domain using spectral convolutions:

* Input → $1 \times 1$ lifting conv → width $=64$
* 3 FNO blocks with truncated modes $(k_1 = k_2 = 16)$
* Each block: spectral conv + residual $1 \times 1$ conv + BatchNorm + GELU
* Global average pooling → $64$-dim feature vector

Key stability fixes:

* Spectral weights stored as real tensors and cast via `torch.view_as_complex`
* FFT executed with AMP disabled to avoid cuFFT FP16 issues
* Input clamped to $[-3, 3]$ to suppress outlier-driven spectral artifacts

---

**CNN branch**
EfficientNet-B0 pretrained on ImageNet:

* First layer adapted to single-channel input
* Final classifier removed
* Outputs a $1280$-dim feature vector

---

**Fusion head**
Concatenated features passed through:

* $1344 \rightarrow 512 \rightarrow 128 \rightarrow 3$
* GELU activations + BatchNorm + Dropout

---

## Motivation

Gravitational lensing signals exist across scales:

* **FNO** captures global structure immediately through frequency-domain representation
* **EfficientNet** captures fine local distortions via spatial convolutions

The hybrid model combines both inductive biases:

* Global spectral consistency (rings, symmetry)
* Local texture sensitivity (subhalo perturbations)

---

## Results

| Model                               | Macro AUC  |
| ----------------------------------- | ---------- |
| EfficientNet-B0 (baseline)          | 0.9744     |
| FNO + EfficientNet (initial, MixUp) | 0.774      |
| FNO + EfficientNet (clean training) | **0.9217** |
| FNO + EfficientNet (final, TTA)     | **0.9439** |

Key observations:

* MixUp limited convergence in later stages
* Removing augmentation and applying input clamping improved stability
* FNO contributes complementary global features but requires careful training

---

## Training Strategy

Training was performed in multiple controlled phases:

**Phase 1–2 (Base training with MixUp)**

* AUC plateau at ~0.92
* MixUp began limiting fine-grained learning

**Phase 3 (Clean fine-tuning)**

* Removed MixUp
* Added input clamping
* Improved to **0.9217**

**Phase 4 (Aggressive augmentation — failed)**

* CutMix + label smoothing destabilized training
* AUC dropped to ~0.84

**Phase 5 (Gradual fine-tuning)**

* Multi-stage LR decay
* No augmentation
* Stable convergence

---

## Training Details

* **Optimizer**: AdamW
* **Scheduler**: OneCycleLR
* **Batch size**: 128
* **Image size**: 64 × 64
* **Hardware**: NVIDIA T4 (~53s/epoch)
* **Normalization**: per-sample mean/std, then clamp to $[-3, 3]$ (FNO path)

---

## Key Insight

The main bottleneck was **spectral instability**, not model capacity.
Unbounded pixel values produced dominant FFT coefficients, masking useful frequency structure.
Clamping the input acted as a physics-informed filter and enabled meaningful learning.

---

## Limitations

* FNO is trained from scratch; no pretrained spectral models exist for this domain
* Hybrid model remains below CNN baseline due to optimization difficulty
* Strong coupling between branches makes aggressive regularization unstable

---

## Future Work

* Apply spectral layers on intermediate CNN features instead of raw input
* Increase FNO width and modes with better memory handling
* Explore physics-informed losses in Fourier space
* Pretrain FNO on synthetic lensing simulations

---

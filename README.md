# Deepfake Audio & Gender Classification: Multi-Modal Dual-Branch Fusion

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-ffea00.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Problem Statement
**Theme:** Hack the Fake, Save the Real Challenge: Deepfake Audio Classification in the Wild.

With the proliferation of generative AI, synthetic audios can convincingly imitate human voices, posing severe threats to digital trust. This repository contains a deep learning pipeline developed for **Task B: Gender classification for Real vs. Fake Voices**. The goal is to accurately distinguish between real and synthetic voices while simultaneously classifying the speaker's gender under real-world, noisy conditions.

---

## 📊 Dataset Overview
The dataset contains a total of 18,003 `.wav` audio files distributed across training and testing directories. The objective is a 4-class classification task.

### Class Mapping (`metaData.csv`)
| Label ID | Class Name | Description |
| :---: | :--- | :--- |
| **0** | `Female_Real` | Female Genuine, unaltered real-world audio recording. |
| **1** | `Male_Real` | Male Genuine, unaltered real-world audio recording. |
| **2** | `Female_Fake` | Synthetic Female audio, potentially distorted or degraded. |
| **3** | `Male_Fake` | Synthetic Male audio, potentially distorted or degraded. |

*Submissions are evaluated based on the **Macro F1-Score** across all four classes.*

---

## 🧠 System Architecture

The core of this solution is a **Dual-Branch Multi-Modal Fusion Network** designed to capture both the high-level semantic context of human speech and the low-level acoustic artifacts typical of AI generation.

### 1. Feature Engineering & Precomputation
To accelerate training and facilitate CNN ingestion, acoustic features are precomputed and cached as PyTorch tensors. 
* **Mel-Spectrograms:** Extracted using 1024 n_fft, 128 Mel bins, and converted to decibel scale.
* **Constant-Q Transform (CQT):** Extracted to capture precise low-frequency harmonics (84 bins). 
* **Alignment:** The CQT output is time-aligned and spatially interpolated to match the Mel-spectrogram dimensions, resulting in a dual-channel tensor: $\mathbf{X}_{\text{spec}} \in \mathbb{R}^{2 \times 128 \times T}$.

### 2. Audio Augmentation Pipeline
To ensure robustness against compression artifacts and background noise ("in the wild" conditions), stochastic augmentations are applied on-the-fly:

**Raw Waveform Augmentations (WavLM Branch):**
* Additive Gaussian Noise ($SNR \in [5, 20]$ dB)
* Random Gain ($-6$ to $+6$ dB)
* Polarity Inversion & Random Clipping
* Time Shifting (Wrap-around shift up to $150$ ms)
* Random Band-Pass Filtering (SoX sinc filters to simulate transmission loss)

**Spectrogram Augmentations (ResNet Branch):**
* SpecAugment (Time masking and Frequency masking) applied across the dual-channel tensors.

### 3. Model Backbones
The architecture leverages two distinct encoders before fusing their representations:

* **Branch A: Audio Transformer (`microsoft/wavlm-base-plus`)**
  * Processes raw, 1D audio arrays.
  * Captures latent phonetic representations and high-level structural anomalies.
  * The last 2 transformer layers are unfrozen for task-specific fine-tuning.
  * Outputs a 768-dimensional embedding vector via mean-pooling.

* **Branch B: Vision CNN (`ResNet50`)**
  * Processes 2D dual-channel spectrograms (Mel + CQT).
  * Captures visual acoustic artifacts (e.g., phase irregularities, high-frequency spectral gaps).
  * The initial convolutional layer is adapted for 2-channel input. The final 2 residual blocks are unfrozen.
  * Outputs a 2048-dimensional embedding via Adaptive Average Pooling.

### 4. Fusion Head

The embeddings from both branches are concatenated along the feature dimension:

$$\mathbf{E}_{\text{fusion}} = \mathbf{E}_{\text{WavLM}} \oplus \mathbf{E}_{\text{ResNet}}$$

The resulting 2816-dimensional vector ($\mathbb{R}^{768 + 2048}$) is passed through a dense classification head:

`Linear(1024)` $\rightarrow$ `ReLU` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(512)` $\rightarrow$ `ReLU` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(4)`

---

## ⚙️ Training Setup & Hyperparameters

* **Loss Function:** `CrossEntropyLoss` combined with **Label Smoothing ($0.1$)** to prevent overconfidence, and **Balanced Class Weights** to mitigate dataset imbalances:
  $$W_c = \frac{N}{C \times n_c}$$
* **Optimizer:** AdamW with a learning rate of $6 \times 10^{-5}$ and weight decay of $0.01$.
* **LR Scheduler:** Cosine Annealing with a $10\%$ linear warmup phase.
* **Precision:** Automatic Mixed Precision (AMP) utilizing `torch.amp.autocast` (`float16`) to drastically reduce VRAM consumption and speed up gradient updates.
* **Custom Early Stopping:** Implements a dual-save mechanism to combat overfitting:
  1. Saves the model conditionally upon a strict **Validation F1** improvement.
  2. Saves an alternate "Best Train" model if Validation F1 plateaus but Train F1 improves.
  * Halts training if no saving conditions are met for 25 consecutive epochs.

---

## 🚀 Installation & Usage

### 1. Environment Setup
Install the necessary dependencies. Note: Ensure you have SoX installed on your system (`sudo apt-get install sox libsox-fmt-all` on Ubuntu) for Torchaudio augmentations to function properly.

```bash
pip install torch torchvision torchaudio transformers librosa soundfile scikit-learn pandas tqdm

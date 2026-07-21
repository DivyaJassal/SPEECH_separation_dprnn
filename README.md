# Multi-Channel Speech Separation using Research-Oriented DPRNN

A research-oriented deep learning framework for **single-channel multi-speaker speech separation** capable of separating **2–4 overlapping speakers** from a single mixed audio recording. The project builds upon the Dual-Path Recurrent Neural Network (DPRNN) architecture and incorporates several research-inspired enhancements to improve separation quality, training stability, and computational efficiency on Apple Silicon devices.

---

## Overview

The cocktail party problem remains one of the fundamental challenges in speech processing, where multiple speakers overlap in a single recording. This project addresses that challenge by implementing a waveform-domain DPRNN-based separator trained using Permutation Invariant Training (PIT) with an SI-SDR optimization objective.

Unlike a standard DPRNN implementation, this repository incorporates research-inspired architectural and training improvements designed to enhance robustness for variable-speaker mixtures while maintaining practical training efficiency.

---

## Key Features

* Single-channel speech separation
* Supports mixtures containing **2–4 speakers**
* End-to-end waveform-domain separation
* DPRNN-based separator
* Permutation Invariant Training (PIT)
* SI-SDR optimization objective
* Variable speaker handling through padded outputs
* Apple Silicon (MPS) optimized training
* Automatic checkpointing
* Evaluation using SI-SDR and SI-SDR Improvement (SI-SDRi)

---

## Research-Oriented Contributions

The baseline DPRNN architecture was extended with several techniques inspired by recent speech separation literature.

### 1. Multi-Scale DPRNN Processing

Multiple temporal chunk sizes are processed during training, enabling the network to capture both short-term phonetic information and long-range conversational dependencies for improved temporal modeling.

### 2. Dynamic Chunk Scheduling

Instead of training with a single fixed chunk size, the model dynamically varies chunk lengths during training, improving robustness to different utterance durations while reducing overfitting.

### 3. Curriculum Learning

Training progressively introduces increasingly difficult mixtures using the dataset's overlap ratio metadata. The model first learns from low-overlap speech mixtures before adapting to highly overlapping conversations, improving convergence stability.

### 4. Improved PIT-Based Optimization

The separation network is trained using Permutation Invariant Training with Scale-Invariant Signal-to-Distortion Ratio (SI-SDR), enabling correct speaker assignment regardless of output ordering.

### 5. Efficient Apple Silicon Training

The implementation is optimized for Apple Silicon devices using the Metal Performance Shaders (MPS) backend, allowing efficient local training without requiring dedicated CUDA hardware.

---

## Model Architecture

```
Mixed Audio
      │
      ▼
Learnable Encoder
      │
      ▼
Dual-Path Chunking
      │
      ▼
Multi-Scale DPRNN Blocks
 ├── Intra-Chunk BiLSTM
 ├── Inter-Chunk BiLSTM
 ├── Residual Connections
 └── Layer Normalization
      │
      ▼
Mask Estimation
      │
      ▼
Waveform Decoder
      │
      ▼
Separated Speaker Signals (2–4 Outputs)
```

---

## Dataset

The project expects the dataset in the following format:

```
conversational_dataset_v2/
│
├── train/
├── val/
└── test/
    ├── sample_000001/
    │   ├── mixture.flac
    │   ├── source_1.flac
    │   ├── source_2.flac
    │   ├── source_3.flac
    │   ├── source_4.flac
    │   └── metadata.json
```

Each sample contains:

* Mixed waveform
* Individual speaker waveforms
* Speaker metadata
* Overlap ratio
* Number of active speakers

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Multi_channel_audio
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train the DPRNN model:

```bash
python3 src/train.py
```

The best model checkpoint is automatically saved to:

```
checkpoints/best_dprnn.pt
```

---

## Evaluation

Evaluate the trained model:

```bash
python3 src/evaluate.py
```

Generated outputs include:

* `reports/results_test.csv`
* `reports/results_test.png`

Evaluation metrics:

* SI-SDR
* SI-SDR Improvement (SI-SDRi)
* Per-speaker statistics
* Overall average performance

---

## Separate New Audio

Separate an unseen mixture:

```bash
python3 src/separate.py \
    --checkpoint checkpoints/best_dprnn.pt \
    --input path/to/mixture.flac \
    --output outputs/
```

---

## Configuration

All hyperparameters are configurable through:

```
configs/config.yaml
```

Key configurable parameters include:

* Batch size
* Learning rate
* Number of DPRNN blocks
* Hidden dimensions
* Encoder dimensions
* Chunk sizes
* Segment duration
* Number of workers
* Validation interval

---

## Project Structure

```
Multi_channel_audio/
│
├── configs/
├── conversational_dataset_v2/
├── checkpoints/
├── reports/
├── outputs/
├── src/
│   ├── dataset.py
│   ├── dataloader.py
│   ├── dprnn.py
│   ├── losses.py
│   ├── trainer.py
│   ├── evaluate.py
│   ├── separate.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

---

## Results

The model is evaluated using objective speech separation metrics:

* **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)**
* **SI-SDR Improvement (SI-SDRi)**

Training automatically saves the best-performing checkpoint based on validation loss, while the evaluation pipeline generates detailed CSV reports and visualization plots.

---

## Future Improvements

* Learnable speaker query embeddings
* Speaker presence gating
* Deep supervision across DPRNN blocks
* Multi-resolution STFT auxiliary loss
* Mixed precision training
* Distributed multi-GPU training
* ONNX/TorchScript deployment

---

## Technologies Used

* Python
* PyTorch
* Torchaudio
* DPRNN
* BiLSTM
* Permutation Invariant Training (PIT)
* SI-SDR
* Apple Silicon MPS
* Pandas
* Matplotlib
* TQDM

---

## License

This project is intended for academic research and educational purposes.

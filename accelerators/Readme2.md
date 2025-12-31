
# LLM Inference Workshop 

This document explains how to log into the Jetson Xavier device, activate the Python environment, and locate the provided LLM inference scripts used in the workshop.

---

## 1️ SSH into the Jetson Device

Connect to the Jetson as `edgeuser` using the following IP address:

```bash
ssh edgeuser@10.24.24.60
ssh edgeuser@10.24.24.63
```

## Navigate to the Workshop Directory

```bash
cd /media/ssd/workshop
```
## Activate the Python Virtual Environment

```bash
source venv/bin/activate
```

## Verify Workshop Files

```bash
ls
```

You should see the following files and folders:

```bash
dataset.py
dataset/
venv/
weights/
lenet5.py
train_lenet.py
infer_lenet.py
bench.py
benchmarks/
vision/
```

## File Overview

* **`dataset.py`**
  Defines dataset loading and preprocessing utilities. Typically wraps MNIST (or a custom dataset) and standardizes transforms, splits, and access patterns used consistently across training, inference, and benchmarking.

* **`dataset/`**
  Local data directory containing downloaded or prepared datasets (e.g., MNIST images and labels). Acts as the persistent storage location to avoid repeated downloads, especially important on edge devices with limited bandwidth.

* **`venv/`**
  Python virtual environment containing PyTorch, torchvision, and all required dependencies. Ensures reproducible execution across training, inference, and benchmarking runs.

* **`weights/`**
  Stores trained model checkpoints (e.g., `lenet5_mnist.pth`). Used as the handoff point between training (`train_lenet.py`) and inference/benchmarking (`infer_lenet.py`, `bench.py`).

* **`lenet5.py`**
  Contains the LeNet-5 model definition, including convolutional layers, fully connected layers, and forward pass logic. Serves as the single source of truth for the network architecture.

* **`train_lenet.py`**
  Handles end-to-end training of the LeNet-5 model:

  * Dataset loading
  * Model initialization
  * Loss and optimizer setup
  * Training loop and checkpoint saving

* **`infer_lenet.py`**
  Runs inference using a trained LeNet-5 checkpoint. Designed for validating model correctness, measuring single-sample latency, or integrating into downstream pipelines.

* **`bench.py`**
  Executes performance benchmarks for the model, typically measuring:

  * Inference latency
  * Throughput (images/sec)
  * CPU vs GPU execution characteristics
    Useful for edge-device performance analysis.

* **`benchmarks/`**
  Contains benchmark artifacts such as logs, CSV results, or plots (e.g., latency vs batch size). Enables reproducibility and comparison across runs, devices, or configurations.


Below is a **complete, README-ready “Running a Script” section** that explicitly covers **all relevant Python entry points** in your repository, written in the same style and level of detail as your LLM example.

---

## Running a Script

### Dataset Preparation

Initializes and verifies dataset loading and preprocessing logic.

```bash
python3 dataset.py
```

---

### Train the Model

Trains the LeNet-5 model on the configured dataset and saves the learned weights to the `weights/` directory.

```bash
python3 train_lenet.py
```

---

### Run Inference

Loads a trained LeNet-5 checkpoint and performs inference on sample inputs for validation or deployment testing.

```bash
python3 infer_lenet.py
```

---

### Run Benchmarks

Runs performance benchmarks to evaluate inference latency and throughput across different execution configurations.

```bash
python3 bench.py
```
---

### Environment Setup
Run this in your loacl machine

```bash
git clone <REPO_URL>
cd <REPO_NAME>
```

Run the following command from the repository root:

```bash
uv sync
```

###Local Visual Benchmarks

```bash
scp -r edgeuser@10.24.24.63:/media/ssd/workshop/benchmarks .
```

```bash
uv run bench_plots.py
```

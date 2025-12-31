import torch
import time
import csv
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lenet5 import LeNet5

# =========================
# Configuration
# =========================
DATA_ROOT = "/media/ssd/workshop/dataset"
OUT_DIR = Path("./benchmarks")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "lenet5_benchmark.csv"

BATCH_SIZES = [32, 64, 128, 256]
LR = 1e-3
EPOCHS = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# =========================
# Dataset
# =========================
transform = transforms.ToTensor()

train_ds = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=transform
)

test_ds = datasets.MNIST(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=transform
)

# =========================
# CSV Setup
# =========================
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "batch_size",
        "device",
        "epoch_time_sec",
        "train_throughput_samples_sec",
        "inference_latency_ms",
        "inference_throughput_samples_sec",
        "accuracy",
        "max_gpu_mem_mb"
    ])

# =========================
# Benchmark Loop
# =========================
for batch_size in BATCH_SIZES:
    print(f"\n=== Benchmarking batch size {batch_size} ===")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda")
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda")
    )

    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # -------------------------
    # Training Benchmark
    # -------------------------
    model.train()
    start = time.time()

    for _ in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    epoch_time = time.time() - start
    train_throughput = len(train_loader.dataset) / epoch_time

    # -------------------------
    # Inference Latency (1 sample)
    # -------------------------
    model.eval()
    x, _ = next(iter(test_loader))
    x = x[:1].to(device)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    latency_ms = (time.time() - start) * 1000

    # -------------------------
    # Inference Throughput (batch)
    # -------------------------
    x, _ = next(iter(test_loader))
    x = x.to(device)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    infer_time = time.time() - start
    infer_throughput = x.size(0) / infer_time

    # -------------------------
    # Accuracy
    # -------------------------
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total

    # -------------------------
    # Memory
    # -------------------------
    max_mem_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if device == "cuda"
        else 0
    )

    # -------------------------
    # Save Row
    # -------------------------
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            batch_size,
            device,
            round(epoch_time, 4),
            round(train_throughput, 2),
            round(latency_ms, 3),
            round(infer_throughput, 2),
            round(accuracy, 4),
            round(max_mem_mb, 2)
        ])

    print("✓ Done")

print(f"\n📊 Benchmark data saved to {CSV_PATH}")

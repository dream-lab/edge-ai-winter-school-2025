import torch
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from lenet5 import LeNet5

# -------------------------
# Paths
# -------------------------
DATA_ROOT = "/media/ssd/workshop/dataset"
WEIGHTS_PATH = Path("/media/ssd/workshop/weights/lenet5_mnist.pth")

# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet5().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

test_loader = DataLoader(
    datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=256,
    shuffle=False
)

# -------------------------
# Warm-up
# -------------------------
with torch.no_grad():
    x, _ = next(iter(test_loader))
    model(x.to(device))

torch.cuda.synchronize()

# -------------------------
# Inference + timing
# -------------------------
start = time.time()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

torch.cuda.synchronize()
end = time.time()

# -------------------------
# Results
# -------------------------
print(f"Accuracy      : {correct / total:.4f}")
print(f"Total samples : {total}")
print(f"Time (s)      : {end - start:.3f}")
print(f"Throughput    : {total / (end - start):.2f} samples/sec")


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from lenet5 import LeNet5

# -------------------------
# Paths
# -------------------------
DATA_ROOT = "/media/ssd/workshop/dataset"
WEIGHTS_DIR = Path("/media/ssd/workshop/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = WEIGHTS_DIR / "lenet5_mnist.pth"

# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet5().to(device)

train_loader = DataLoader(
    datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=64,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# -------------------------
# Train (1 epoch is enough)
# -------------------------
model.train()
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    loss_fn(model(x), y).backward()
    optimizer.step()

# -------------------------
# Save weights
# -------------------------
torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"✅ Weights saved to: {WEIGHTS_PATH}")


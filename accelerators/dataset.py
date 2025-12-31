from torchvision import datasets, transforms
from pathlib import Path

DATA_ROOT = Path("/media/ssd/workshop/dataset")

def get_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_set = datasets.MNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform
    )

    return test_set


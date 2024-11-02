"""File for dataloaders and data processing functions."""

from pathlib import Path
from typing import TypeVar

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, MNIST, VisionDataset  # type: ignore
from torchvision.transforms import v2  # type: ignore

from experiments.base import RESOURCES, SEED, logger

ROOT_PATH = RESOURCES / "data"

T = TypeVar("T")


def MNISTDataset(
    root_path: Path = ROOT_PATH,
) -> tuple[MNIST, ...]:
    """Get training and test set from MNIST"""

    # Transform data
    t = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # scale to [0, 1]
            torch.flatten,
        ]
    )

    root = str(root_path)
    train_data = MNIST(root=root, train=True, download=True, transform=t)
    logger.info("Train size: %s", len(train_data))
    test_data = MNIST(root=root, train=False, download=True, transform=t)
    logger.info("Test size: %s", len(test_data))

    return train_data, test_data


def train_val_dataset_split(
    dataset: VisionDataset, val_size: int = 10000, seed: int = SEED
) -> tuple[Subset, ...]:
    """Return train - validation split given a torch dataset"""
    # Create a seeded rng to determinstically get a validation set
    rng = torch.Generator().manual_seed(seed)

    train_size, val_size = len(dataset) - val_size, val_size

    train_set, val_set = random_split(dataset, (train_size, val_size), generator=rng)
    logger.info("Train set: %s", len(train_set))
    logger.info("Valid set: %s", len(val_set))
    return train_set, val_set


def get_cifar(root_path: Path = ROOT_PATH, subset: int = 10):

    # Create a seeded rng to determinstically get a validation set
    rng = torch.Generator().manual_seed(SEED)

    IMAGE_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    IMAGENET_TRANSFORM = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # scale to [0, 1]
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    cifar_trainset = CIFAR10(
        root=root_path, train=True, download=True, transform=IMAGENET_TRANSFORM
    )
    cifar_testset = CIFAR10(
        root=root_path, train=False, download=True, transform=IMAGENET_TRANSFORM
    )

    if subset > 1:
        cifar_trainset = Subset(
            cifar_trainset, indices=range(0, len(cifar_trainset), subset)
        )
        cifar_testset = Subset(
            cifar_testset, indices=range(0, len(cifar_testset), subset)
        )

    cifar_trainset, cifar_valset = random_split(
        cifar_trainset, [0.9, 0.1], generator=rng
    )

    return cifar_trainset, cifar_valset, cifar_testset


def get_mnist(train_subset: int = 1, train_val_split=[0.9, 0.1]):
    """Return train, validation and test sets for MNIST.

    ~~Normalisation: mean 0, std 1~~
    Normalisation: [0,1] data is sufficient for MNIST.
    Might work better for CNNs as the padding is 0, which is the mode.

    Args:
        subset: Get every `subset`-th element from the train dataset.
        Equivalently, the fraction of the dataset to use is 1 / `subset`.
    """

    # Create a seeded rng to determinstically get a validation set
    rng = torch.Generator().manual_seed(SEED)

    t = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # scale to [0, 1]
            # v2.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )

    root = str(ROOT_PATH)
    train_data = MNIST(root=root, train=True, download=True, transform=t)
    test_data = MNIST(root=root, train=False, download=True, transform=t)

    logger.info("Train size: %s", len(train_data))
    logger.info("Test size: %s", len(test_data))

    train_data, val_data = random_split(train_data, train_val_split, generator=rng)
    if train_subset > 1:
        train_data = Subset(train_data, indices=range(0, len(train_data), train_subset))

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_set, test_set = MNISTDataset()
    train_set, val_set = train_val_dataset_split(train_set, val_size=10000)

    trainset, valset, testset = get_cifar()

    trainset, valset, testset = get_mnist()
    breakpoint()
    print("Done")

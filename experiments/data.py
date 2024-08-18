"""File for dataloaders and data processing functions."""

from typing import TypeVar
from experiments.base import RESOURCES, SEED, logger

import torch
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import v2  # type: ignore
from torch.utils.data import DataLoader, random_split, Subset, Dataset


from pathlib import Path

T = TypeVar("T")


def MNISTDataset(
    root_path: Path = RESOURCES / "data",
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
    dataset: Dataset[T], val_size: int = 10000, seed: int = SEED
) -> tuple[Subset[T], ...]:
    """Return train - validation split given a torch dataset"""
    # Create a seeded rng to determinstically get a validation set
    rng = torch.Generator().manual_seed(seed)

    train_size, val_size = len(dataset) - val_size, val_size

    train_set, val_set = random_split(dataset, (train_size, val_size), generator=rng)
    logger.info("Train set: %s", len(train_set))
    logger.info("Valid set: %s", len(val_set))
    return train_set, val_set

"""Implement a light configurable CNN """

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from experiments.base import BaseConfig, BaseModel


@dataclass
class CNNConfig(BaseConfig):
    in_channels: int = 1
    img_size: int = 28  # 1 x 28 x 28 image
    hidden_features: int = 16
    n_classes: int = 10


class CNN(BaseModel):
    """Lightweight CNN"""

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config

        self.nnet = nn.Sequential(
            nn.Conv2d(
                self.config.in_channels,
                self.config.hidden_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(self.config.hidden_features),
        )
        # TODO: padding should be equal to the "zeros" on the border
        # But this may not be zero anymore due to the normalisation

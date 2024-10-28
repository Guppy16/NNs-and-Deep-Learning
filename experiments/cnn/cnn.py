"""Implement a light configurable CNN """

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from experiments.base import BaseConfig, BaseModel


@dataclass
class CNNConfig(BaseConfig):
    out_features: int = 16
    n_classes: int = 10


class CNN(BaseModel):
    def __init__(self, config: CNNConfig):
        """"""
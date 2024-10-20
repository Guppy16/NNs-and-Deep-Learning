"""Implementation of ResNet"""

import torch
from torch import nn, Tensor

import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch.nn as nn
from jaxtyping import Float, Int
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
import torch.nn.functional as F

from experiments.base import copy_state_dict, BaseConfig, BaseModel
from dataclasses import dataclass, field


@dataclass
class ResNet34Config(BaseConfig):
    n_blocks_per_group: list[int] = field(default_factory=lambda: [3, 4, 6, 3])
    out_features_per_group: list[int] = field(
        default_factory=lambda: [64, 128, 256, 512]
    )
    first_strides_per_group: list[int] = field(default_factory=lambda: [1, 2, 2, 2])
    n_classes: int = 1000


class AveragePool(nn.Module):
    """Perform average pooling over the height and width dims"""

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c"]:
        return torch.mean(x, dim=(2, 3))


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model,
        declare the left side branch first.

        If first_stride is > 1, this means the optional (conv + bn)
        should be present on the right branch.
        """
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride
        self.is_downsampling = first_stride != 1
        self.kernel_size = 3

        self.left = nn.Sequential(
            nn.Conv2d(
                in_feats,
                out_feats,
                self.kernel_size,
                stride=first_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_feats),
            nn.modules.ReLU(),
            nn.Conv2d(out_feats, out_feats, self.kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_feats),
        )

        if not self.is_downsampling:
            assert in_feats == out_feats
            self.right = nn.Identity()
        else:
            self.right = nn.Sequential(
                nn.Conv2d(
                    in_feats,
                    out_feats,
                    kernel_size=1,
                    stride=self.first_stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_feats),
            )
        self.act = nn.modules.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present,
        the addition should just add the left branch's output to the input.
        """
        left = self.left(x)
        right = self.right(x)
        x = self.act(left + right)
        return x


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride."""
        super().__init__()
        self.n_blocks = n_blocks
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        self.block_group = nn.Sequential(
            ResidualBlock(in_feats, out_feats, first_stride),
            *[ResidualBlock(out_feats, out_feats, 1) for _ in range(n_blocks - 1)],
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.block_group(x)


class ResNet34(BaseModel):
    def __init__(self, config: ResNet34Config):
        """ResNet34 implementation

        NOTE: input channels is 3, because we have RGB img
        """
        super().__init__()
        self.config = config

        first_layer_feats = 64
        in_feat_blocks = [first_layer_feats] + config.out_features_per_group
        out_feat_blocks = config.out_features_per_group

        self.nnet = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=first_layer_feats,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=first_layer_feats),
            nn.modules.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *[
                BlockGroup(block, in_feat, out_feat, first_stride)
                for block, in_feat, out_feat, first_stride in zip(
                    config.n_blocks_per_group,
                    in_feat_blocks,
                    out_feat_blocks,
                    config.first_strides_per_group,
                )
            ],
            AveragePool(),
            nn.Linear(config.out_features_per_group[-1], config.n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        return self.nnet(x)

    @torch.inference_mode()
    def inference(self, x: Float[Tensor, "batch channels height width"]):
        """Get the predicted class and probability"""
        self.eval()
        logits = self(x)  # batch x n_classes
        probs = F.softmax(logits, dim=-1)  # batch x n_classes
        idxs = torch.argmax(probs, dim=-1)  # batch
        confidence = probs[idxs]

        return idxs, confidence


def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
    """Returns ResNet34 instance with replaced final linear layer.

    Loads in pretrained weights: `ResNet34_Weights.IMAGENET1K_V1`
    The model weights are frozen, except for the final layer.
    """
    config = ResNet34Config()
    resnet = ResNet34(config)

    # Load in pretrained weights
    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    resnet, _ = copy_state_dict(pretrained_resnet, resnet)
    assert isinstance(resnet, ResNet34)

    # Recursively freeze _all_ layers
    resnet.requires_grad_(False)

    # Replace final layer
    linear = nn.Linear(resnet.config.out_features_per_group[-1], n_classes)
    resnet.nnet[-1] = linear

    # Modify the class config
    resnet.config.n_classes = n_classes

    return resnet


if __name__ == "__main__":
    config = ResNet34Config()
    my_resnet = ResNet34(config)
    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    my_resnet, _ = copy_state_dict(pretrained_resnet, my_resnet)

    resnet = get_resnet_for_feature_extraction(10)
    # print(list(resnet.named_parameters()))
    print("Done")

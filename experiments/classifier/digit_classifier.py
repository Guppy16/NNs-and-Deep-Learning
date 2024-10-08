"""Module to load MNIST dataset and a configurable NN

Some examples:
>>> import ...
"""

import logging
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import itertools
from tqdm import tqdm
import time
import sys
from typing import TypedDict
from dataclasses import dataclass, field
from pathlib import Path
from dataclass_wizard import YAMLWizard
import pickle
import os

from experiments.base import logger, DEVICE, RESOURCES, SEED, set_seeds
from experiments.data import (
    MNISTDataset as train_test_datasets,
    train_val_dataset_split,
)

dir_path = os.path.dirname(os.path.realpath(__file__))


@dataclass
class Metrics:
    """Keep track of metrics while training
    Args:
      train_batch_loss: train loss after each batch
      train_precision: precision (accuracy) on train set after each epock
      valid_precision: precision (accuracy) on validation set after each epoch
    """

    train_batch_loss: list[float] = field(default_factory=list)
    train_precision_epoch: list[float] = field(default_factory=list)
    valid_precision_epoch: list[float] = field(default_factory=list)


@dataclass
class DigitClassifierConfig(YAMLWizard):
    """Config dataclass for DigitClassifier NN

    Args:
      sizes: size of each layer
      learning_rate: learning rate when optimising parameters
      loss: loss module used for calulating loss
      device: torch device type (cuda or cpu)
      mini_batch: size of mini batch on training and validation set
    """

    sizes: tuple[int, ...]
    learning_rate: float
    # loss: nn.modules.loss.MSELoss
    loss: nn.Module
    device: torch.device = DEVICE
    mini_batch: int = 10


class DigitClassifier(nn.Module):
    # Some constants to define how the model and params are stored
    MODEL_FILENAME = "model.tar"
    CONFIG_FILENAME = "config.pkl"

    def __init__(
        self,
        config: DigitClassifierConfig,
    ):
        """
        Args:
          config: DigitClassifier config parameters
        """
        super().__init__()
        self.config = config
        self.num_layers = len(self.config.sizes)
        self.act_fn = nn.Sigmoid()
        # Define linear weights between each layer:
        self.linears = nn.ModuleList(
            [
                nn.Linear(ip, op)
                for ip, op in zip(self.config.sizes, self.config.sizes[1:])
            ]
        )

        self.num_classes = self.config.sizes[-1]
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.config.learning_rate
        )
        self.loss_module = self.config.loss

        # Load train and validation data
        train_data, _ = train_test_datasets()
        train_set, val_set = train_val_dataset_split(train_data)
        self.train_dataloader = DataLoader(
            train_set,
            batch_size=self.config.mini_batch,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        self.val_dataloader = DataLoader(
            val_set,
            batch_size=self.config.mini_batch,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        # Set device
        self.to(self.config.device)

    def save_model(self, model_dir: Path) -> tuple[str, str]:
        """Save model and config to model_path and return file locations"""
        model_dir.mkdir(parents=True, exist_ok=True)
        # Save model
        model_path = model_dir / DigitClassifier.MODEL_FILENAME
        torch.save(self.state_dict(), model_path)

        # Save params
        self.config.to_yaml_file(str(model_dir / "config.yaml"))
        config_path = model_dir / DigitClassifier.CONFIG_FILENAME
        with open(config_path, "wb") as f:
            pickle.dump(self.config, f)

        return model_path, config_path

    @staticmethod
    def load_model(model_dir: Path) -> "DigitClassifier":
        """Load model from model_path and return an instance of the model"""
        if not model_dir.is_dir():
            raise FileNotFoundError
        # Load config
        with open(model_dir / DigitClassifier.CONFIG_FILENAME, "rb") as f:
            config = pickle.load(f)
        # config = DigitClassifierConfig.from_yaml_file(str(config_path))
        model = DigitClassifier(config)
        # Load state dict
        model_path = model_dir / DigitClassifier.MODEL_FILENAME
        state_dict = torch.load(str(model_path))
        model.load_state_dict(state_dict)
        return model

    def forward(self, x):
        """Forward pass over all neurons"""
        for layer in self.linears:
            x = layer(x)
            x = self.act_fn(x)
        return x

    def train_loop(
        self,
        *,
        metrics: Metrics,
        num_epochs: int = 30,
        func: callable = None,
        **func_kwargs,
    ):
        """Training loop for NN

        Args:
          metrics: see Metrics
          num_epochs: number of iterations over whole dataset
          func: function to run after each epoch. Takes model as arg
          **func_kwargs: kwargs to pass to func
        """
        # Set model to train mode
        self.train()

        # Training loop
        for epoch in range(num_epochs):
            # Before each epoch
            # Run func
            if func:
                func(self, **func_kwargs)
            for data_inputs, data_labels in self.train_dataloader:
                # 1. Move input data to device
                data_inputs = data_inputs.to(DEVICE)
                data_labels = data_labels.to(DEVICE)

                # 2. Run model on input data
                preds = self.forward(data_inputs)

                # 3. Calculate loss
                # ? TODO: Abstract this to give options based on loss module
                loss = self.loss_module(
                    preds,
                    nn.functional.one_hot(
                        data_labels, num_classes=self.num_classes
                    ).float(),
                )
                metrics.train_batch_loss.append(float(loss.detach().cpu()))

                # 4. Perform backpropogation
                self.optimizer.zero_grad()
                loss.backward()

                # 5. Update parameters
                self.optimizer.step()

            # After epoch
            # Evaluate model on training set
            precision = self.precision(self.train_dataloader)
            metrics.train_precision_epoch.append(float(precision.cpu()))
            # Evaluate model on validation set
            precision = self.precision(self.val_dataloader)
            metrics.valid_precision_epoch.append(float(precision.cpu()))
            # Log accuracy
            total = len(self.val_dataloader.dataset)
            logger.info(f"Epoch: {epoch}: {precision*total:.0f} / {total}")
            # TODO: Calculate loss on validation set

    @torch.no_grad()
    def precision(self, data_loader: DataLoader):
        """Return precision of data"""
        self.eval()
        predicted, labels = zip(*((self.forward(ip), lbl) for ip, lbl in data_loader))
        # Stack batched (predictions, labels) and reshape
        predicted = torch.stack(predicted).reshape(-1, self.num_classes)
        labels = torch.stack(labels).flatten()
        # Use argmax to calculate predicted labels
        predicted = predicted.argmax(-1)

        # Calculate true positives
        true_positives = (predicted == labels).sum()

        return true_positives / len(labels)

    def layer_weights(self, from_layer: int, to_next_layer_node: int) -> Tensor:
        """Return detached weights from a layer to a node in the next layer"""

        if from_layer > self.num_layers:
            raise ValueError(f"Model has {self.num_layers} but got {from_layer}")
        weights = self.linears[from_layer].weight
        if to_next_layer_node > len(weights):
            raise ValueError(
                f"Next layer has {len(weights)} nodes bu got {to_next_layer_node}"
            )
        return weights[to_next_layer_node].detach().cpu()


def base() -> None:
    """Run some default configurations"""
    set_seeds()


if __name__ == "__main__":
    base()
    model_config = DigitClassifierConfig(
        sizes=[784, 30, 10],
        learning_rate=3,
        device=DEVICE,
        loss=nn.MSELoss(reduction="mean"),
        mini_batch=10000,
    )
    model = DigitClassifier(model_config)

    # Save model
    model_path = RESOURCES / "model" / "test"
    model.save_model(model_path)

    # Load model
    m = DigitClassifier.load_model(model_path)
    # assert m == model # This doesn't work for some reason

    epochs = 1
    metrics = Metrics()
    model.train_loop(num_epochs=epochs, metrics=metrics)

    logger.info("Train loss metrics: %s", metrics.train_batch_loss)
    logger.info("Validation accuracy metrics: %s", metrics.valid_precision_epoch)

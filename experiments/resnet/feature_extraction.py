from dataclasses import dataclass
from typing import Type
import torch
from torch import Tensor
from torch.utils.data import Subset, DataLoader

from experiments.resnet.resnet import get_resnet_for_feature_extraction
from experiments.base import (
    set_seeds,
    RESOURCES,
    TBLogger,
    logger,
)
from datetime import datetime
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
from experiments.data import get_cifar

from jaxtyping import Float, Int


@dataclass
class ResNetTrainingArgs:
    batch_size: int = 64
    epochs: int = 3
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    learning_rate: float = 1e-3
    n_classes: int = 10
    subset: int = 10


def loss_function(
    logits: Float[Tensor, "batch n_classes"],
    labels: list[int],
    tb_logger: TBLogger,
    tb_log_to="Loss/train",
):
    """Cross-Entropy Loss on the model outputs compared against the batch"""
    pass


def train_loop(
    model: torch.nn.Module,
    trainset: Subset,
    valset: Subset,
    training_args: ResNetTrainingArgs,
    device: torch.device,
    tb_logger: TBLogger,
):
    set_seeds()

    train_loader = DataLoader(
        trainset, batch_size=training_args.batch_size, shuffle=True
    )
    val_loader = DataLoader(valset, batch_size=training_args.batch_size, shuffle=False)

    model = model.to(device)
    optimizer = training_args.optimizer(
        model.parameters(), lr=training_args.learning_rate
    )

    global_step = 0

    for epoch in tqdm(range(training_args.epochs), desc="Epochs"):
        epoch_train_loss = 0.0
        with tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            unit="batch",
            desc=f"Epoch {epoch + 1}",
            disable=False,
        ) as pbar:

            for batch_idx, (x, labels) in pbar:
                # 0. Set model to training mode
                model.train()
                # 1. Move input to device
                x = x.to(device)
                # 2. Zero the gradients
                optimizer.zero_grad()
                # 3. Forward pass
                logits = model(x)
                # 4. Compute loss
                loss = loss_function(logits, labels, tb_logger)
                # 5. Backward pass
                loss.backward()
                # 6. Update weights
                optimizer.step()
                # Update progress bar
                loss_item = loss.detach().cpu().item()
                epoch_train_loss += loss_item
                tb_logger.add_scalar("Loss/train", loss_item, global_step)
                pbar.set_postfix({"Loss": loss_item})
                global_step += 1

        logger.info(f"Training Loss per item: {epoch_train_loss / len(train_loader)}")

        if val_loader is not None:
            model.eval()
            val_loss = 0
            for x, labels in val_loader:
                x = x.to(device)
                loss = loss_function(logits, labels, tb_logger, tb_log_to="Loss/val")
                val_loss += loss.detach().cpu().item()
            logger.info(f"Validation Loss: {val_loss / len(val_loader)}")
            tb_logger.add_scalar("Loss/val", val_loss / len(val_loader), global_step)


if __name__ == "__main__":

    training_args = ResNetTrainingArgs(epochs=1)

    model_dir = (
        RESOURCES
        / "models"
        / "resnet"
        / (datetime.now().strftime("%Y%m%d-%H%M%S") + "lr1e3")
    )

    trainset, valset, testset = get_cifar()
    resnet = get_resnet_for_feature_extraction(n_classes=training_args.n_classes)
    tb_logger = TBLogger(model_dir / "logs")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loop(resnet, trainset, valset, training_args, device, tb_logger)

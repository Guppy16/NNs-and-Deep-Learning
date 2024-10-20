from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Type

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from experiments.base import RESOURCES, TBLogger, logger, set_seeds
from experiments.data import get_cifar
from experiments.resnet.resnet import get_resnet_for_feature_extraction


@dataclass
class ResNetTrainingArgs:
    batch_size: int = 64
    epochs: int = 3
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    learning_rate: float = 1e-3
    n_classes: int = 10
    subset: int = 10


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
                labels = labels.to(device)
                # 2. Zero the gradients
                optimizer.zero_grad()
                # 3. Forward pass
                logits = model(x)
                # 4. Compute loss
                loss = F.cross_entropy(logits, labels, reduction="mean")
                # 5. Backward pass
                loss.backward()
                # 6. Update weights
                optimizer.step()
                # Update progress bar
                loss_item = loss.detach().cpu().item()
                epoch_train_loss += loss_item  # loss per item
                tb_logger.add_scalar("Loss/train", loss_item, global_step)
                pbar.set_postfix({"Loss": loss_item})
                global_step += 1

        epoch_train_loss = epoch_train_loss / len(train_loader)  # loss per item
        logger.info(f"Training Loss per item: {epoch_train_loss}")

        if val_loader is not None:
            model.eval()
            val_loss = 0
            for x, labels in val_loader:
                x = x.to(device)
                classes, _ = model.inference(x)
                classes = classes.detach().cpu()
                # prediction loss
                loss = (classes == labels).sum().item()
                val_loss += loss
            val_loss = val_loss / len(valset)
            logger.info("Validation Accuracy: %s", val_loss)
            tb_logger.add_scalar("Accuracy/val", val_loss, global_step)


if __name__ == "__main__":

    # training_args = ResNetTrainingArgs(epochs=1)
    training_args = ResNetTrainingArgs(epochs=3)

    model_dir = (
        RESOURCES
        / "models"
        / "resnet"
        / (datetime.now().strftime("%Y%m%d-%H%M%S") + "lr1e3")
    )

    # trainset, valset, testset = get_cifar(subset=100)
    trainset, valset, testset = get_cifar(subset=10)
    resnet = get_resnet_for_feature_extraction(n_classes=training_args.n_classes)
    tb_logger = TBLogger(model_dir / "logs")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loop(resnet, trainset, valset, training_args, device, tb_logger)

"""Training script for CNN

"""

from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.base import RESOURCES, BaseConfig, TBLogger, logger, set_seeds
from experiments.cnn.cnn import CNN, CNNConfig
from experiments.data import get_mnist


@dataclass
class CNNMnistTrainerConfig(BaseConfig):
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 1e-3
    subset: int = 1
    width: int = 16  # hidden_features in CNN


class CNNMnistTrainer:
    def __init__(self, config: CNNMnistTrainerConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CNN(CNNConfig(hidden_features=self.config.width))
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), self.config.learning_rate
        )
        self.trainset, self.valset, self.testset = get_mnist(
            train_subset=self.config.subset
        )

        model_dir = (
            RESOURCES
            / "models"
            / "cnn"
            / (datetime.now().strftime("%Y%m%d-%H%M%S") + "")
        )

        self.tb_logger = TBLogger(model_dir / "logs")

    def training_step(
        self, imgs: Float[Tensor, "b c h w"], labels: Int[Tensor, "b"]
    ) -> Float[Tensor, "b"]:
        """CE Loss on labels"""

        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    @torch.inference_mode()
    def validation_step(self, imgs: Float[Tensor, "b c h w"], labels: Int[Tensor, "b"]):
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(imgs)  # b x classes
        predictions = logits.argmax(dim=-1)
        predictions = predictions == labels
        val_accuracy = predictions.sum().item()
        return val_accuracy

    def train(self):
        global_step = 0
        for epoch in range(self.config.epochs):
            train_loader = DataLoader(
                self.trainset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,
            )
            val_loader = DataLoader(
                self.valset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2,
            )

            self.model.train()
            # self.optimizer.zero_grad()

            with tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                unit="batch",
                desc=f"Epoch {epoch + 1}",
                disable=False,
            ) as pbar:
                for _, (imgs, labels) in pbar:
                    loss = self.training_step(imgs, labels)
                    loss = loss.detach().cpu().item()
                    self.tb_logger.add_scalar("Loss/train/ce", loss, global_step)
                    pbar.set_postfix({"Loss": loss})

                    global_step += 1

            self.model.eval()
            accuracy = sum(
                self.validation_step(imgs, labels) for imgs, labels in val_loader
            )
            accuracy = accuracy / len(self.valset)  # NOTE: valset, not val_loader!!
            self.tb_logger.add_scalar("Accuracy/val", accuracy)
            logger.info("Validation Accuracy:\t%s", accuracy)


if __name__ == "__main__":
    config = CNNMnistTrainerConfig()
    trainer = CNNMnistTrainer(config)
    trainer.train()
    breakpoint()

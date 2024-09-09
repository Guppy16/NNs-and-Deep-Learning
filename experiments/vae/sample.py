"""Sample from the latent space to generate new images using the trained VAE model."""

from pathlib import Path
import torch
import torch.nn as nn

from tqdm import tqdm
from experiments.base import (
    logger,
    DEVICE,
    set_seeds,
    RESOURCES,
    TBLogger,
    count_parameters,
    BaseConfig,
)
from omegaconf import OmegaConf
from dataclasses import dataclass
from torch import Tensor
from torch.optim import AdamW
from experiments.data import MNISTDataset, train_val_dataset_split
from torch.utils.data import DataLoader
from typing import TypeVar
from experiments.vae.vae import VAE
from torchvision.utils import save_image

T = TypeVar("T")


@dataclass
class pCNSamplerConfig(BaseConfig):
    """Config for pCN sampler."""

    model_path: str

    beta: float = 0.1
    num_steps: int = 100


class pCNSampler:

    def __init__(
        self,
        config: pCNSamplerConfig,
        model_type: VAE,
        tb_logger: TBLogger,
    ):
        self.config = config
        self.tb_logger = tb_logger

        # Load model
        self.model = model_type.load_model(self.config.model_path)

        # save sampler config
        self.config.save_config(self.tb_logger.log_dir)

    @torch.no_grad()
    def sample_latent(self, train_loader: DataLoader):
        """Sample using pCN method

        1. Sample from the prior: z ~ N(0, I)
        2. For i = 1 to num_steps:
                1. Sample from pCN proposal
                2. Compute acceptance threshold
                3. Sample u ~ U(0, 1)
                4. If u < threshold, accept proposal, else reject
        3. Save accepted samples

        pCN proposal: z' = sqrt(1 - beta^2)z + beta * epsilon
        epsilon ~ N(0, I)
        acceptance threshold: p(D|z') / p(D|z)
                p(D|z) = prod_i p(x_i|z)

        """

        # keep track of acceptance rate
        accepts = 1  # accept the first sample

        def decode_and_nll(z):
            # Decode the latent
            x_hat = self.model.decoder(z)
            # likelihood of the data given the latent
            nll = 0
            for x, labels in train_loader:
                x = x.to(DEVICE)
                # gaussian negative log likelihood
                gaussian_nll = 0.5 * nn.functional.mse_loss(x_hat, x, reduction="sum")
                nll += gaussian_nll

            return x_hat, nll

        # 1. sample from prior
        z = torch.randn(1, self.model.config.latent_dim).to(DEVICE)
        x_hat, nll = decode_and_nll(z)

        # 2.
        for i in range(self.config.num_steps):
            self.tb_logger.add_scalar("acceptance_rate", accepts / (i + 1), i)
            self.tb_logger.add_scalar("nll", nll, i)
            yield z, x_hat, nll

            # 2.1 sample from proposal
            epsilon = torch.randn(1, self.model.config.latent_dim).to(DEVICE)
            z_prime = (
                torch.sqrt(torch.tensor(1 - self.config.beta**2)) * z
                + self.config.beta * epsilon
            )

            # 2.2 compute acceptance threshold
            # decoded image from the proposed latent
            x_hat_prime, nll_prime = decode_and_nll(z_prime)
            threshold = torch.exp(nll - nll_prime)

            # 2.3 sample u ~ U(0, 1)
            u = torch.rand(1).to(DEVICE)

            # 2.4 accept or reject
            if u < threshold:
                accepts += 1
                z = z_prime
                x_hat = x_hat_prime
                nll = nll_prime


if __name__ == "__main__":
    set_seeds()
    logger.info("Loading config")

    sampler_dir = RESOURCES / "models" / "sampler" / "test"
    sampler_dir.mkdir(parents=True, exist_ok=True)

    model_dir = RESOURCES / "models/vae/20240818-221303l08"

    config = pCNSamplerConfig(model_path=model_dir)
    sampler = pCNSampler(config, VAE, TBLogger(sampler_dir))

    train_set, test_set = MNISTDataset()
    train_set, val_set = train_val_dataset_split(train_set, val_size=10000)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=False, num_workers=4)

    set_seeds()

    images = []
    for z, x_hat, nll in sampler.sample_latent(train_loader):
        images.append(x_hat)

    # save images
    images = torch.cat(images)
    images = images.view(-1, 1, 28, 28)

    save_image(
        images,
        sampler_dir / "images.png",
        # nrow=10,
        # normalize=True,
    )

    logger.info("Saved images to %s", sampler_dir / "images.png")
    breakpoint()

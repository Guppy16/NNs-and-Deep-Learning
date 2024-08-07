from pathlib import Path
from typing import List, TypeVar
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from experiments.base import logger, DEVICE, set_seeds, RESOURCES
from omegaconf import OmegaConf
from dataclasses import dataclass
from torch import Tensor
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

T = TypeVar("T")


@dataclass
class VAEConfig:
    """Config dataclass for VAE NN"""

    # Model parameters
    hidden_dims: tuple[int, ...]  # e.g. (400, 200), (512, 256, 128)
    input_dim: int = 784
    latent_dim: int = 20

    device: str = str(DEVICE)

    batch_size: int = 32

    leaky_relu: float = 0.2
    lr: float = 1e-3

    def save_config(self, dir: Path):
        """Save config to model_dir"""
        conf = OmegaConf.to_yaml(self)
        conf_path = dir / "config.yaml"
        conf_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(conf, conf_path)
        return conf_path

    @staticmethod
    def load_config(fpath: Path) -> "VAEConfig":
        """Load config from file path"""
        config = OmegaConf.load(fpath)
        conf = VAEConfig(**config)
        return conf


class Encoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super(Encoder, self).__init__()

        self.config = config

        if len(self.config.hidden_dims) == 0:
            raise ValueError("Hidden dimensions must be greater than 0")

        modules: List[nn.Module] = []
        in_dim = self.config.input_dim
        for out_dim in self.config.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LeakyReLU(self.config.leaky_relu),
                )
            )
            in_dim = out_dim

        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(self.config.hidden_dims[-1], self.config.latent_dim)
        self.var = nn.Linear(self.config.hidden_dims[-1], self.config.latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.var(x)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super(Decoder, self).__init__()

        self.config = config

        if len(self.config.hidden_dims) == 0:
            raise ValueError("Hidden dimensions must be greater than 0")

        modules: List[nn.Module] = []
        in_dim = self.config.latent_dim
        for out_dim in self.config.hidden_dims[::-1]:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LeakyReLU(self.config.leaky_relu),
                )
            )
            in_dim = out_dim

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            # NOTE: final layer output dim is input dim,
            # because we are reconstructing the input
            nn.Linear(in_dim, self.config.input_dim),
            # NOTE: sigmoid because pixel values are between 0 and 1
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder(x)
        x = self.final_layer(x)

        # NOTE: we are actually outputting a prediction of the "mean" of the input
        return x


class VAE(nn.Module):
    MODEL_FILENAME = "model.tar"
    CONFIG_FILENAME = "config.yaml"

    def __init__(self, config: VAEConfig):
        super(VAE, self).__init__()
        self.config = config
        self.device = torch.device(self.config.device)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.to(self.device)
        self.optimizer = Adam(self.parameters(), lr=self.config.lr)

    def reparemeteraize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick to sample z from N(mu, var) distribution.

        For more stable training, we sample from N(0, 1)
        and scale and shift by mu and var.
        """
        std = torch.exp(0.5 * log_var)  # std = sqrt(var)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def forward(self, x: Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparemeteraize(mu, log_var)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

        # def loss_function(self, x: Tensor, x_hat: Tensor, mu: Tensor, log_var: Tensor):
        #     """VAE loss function.

        #     Maximize the ELBO (Evidence Lower Bound) which is the sum of
        #     Reconstruction loss and KL divergence.

        #     Loss = -ELBO = -E[log P(x|z)] + KL(Q(z|x) || P(z))
        #          = 0.5 * mean|x - x_hat|^2 + 0.5 * mean(-log_var + mu^2 + var) + C

        #     Args:
        #         x: batch x input_dim
        #         x_hat: batch x input_dim
        #         mu: batch x latent_dim
        #         log_var: batch x latent_dim
        #     """

        #     recon_loss = 0.5 * nn.functional.mse_loss(x_hat, x, reduction="mean")

        #     var = torch.exp(log_var)
        #     kld = 0.5 * torch.sum(
        #         var + mu.pow(2) - log_var
        #     )  # Sum over latent dim and batch
        #     kld = kld / self.config.batch_size  # Average over batch

        #     return recon_loss + kld

    def loss_function(self, x, x_hat, mean, log_var):
        """

        x: batch x img_dim, e.g. 32 x 768

        """
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # loss = reproduction_loss + KLD
        loss = reproduction_loss + KLD
        # print("loss", loss.detach().cpu())
        return loss

    def save_model(self, model_dir: Path) -> tuple[Path, Path]:
        """Save model and config to model_path and return file locations"""

        model_dir.mkdir(parents=True, exist_ok=True)

        # save model
        model_path = model_dir / VAE.MODEL_FILENAME
        torch.save(self.state_dict(), model_path)

        # save params
        conf = OmegaConf.to_yaml(self.config)
        conf_path = model_dir / VAE.CONFIG_FILENAME
        with open(conf_path, "w") as f:
            f.write(conf)

        return model_path, conf_path

    @staticmethod
    def load_model(model_dir: Path) -> "VAE":
        """Load model from model_dir and return an instace of the model"""
        if not model_dir.is_dir():
            raise FileNotFoundError

        # load config
        with open(model_dir / VAE.CONFIG_FILENAME, "r") as f:
            config = OmegaConf.load(f)

        model = VAE(config)
        # load state dict
        model_path = model_dir / VAE.MODEL_FILENAME
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        return model

    def train_loop(
        self,
        *,
        epochs: int,
        train_loader: DataLoader[T],
        val_loader: DataLoader[T] | None = None,
    ):
        for epoch in tqdm(range(epochs), desc="Epochs"):
            self.train()
            train_loss = 0
            with tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                unit="batch",
                desc=f"Epoch {epoch + 1}",
                disable=False,
            ) as pbar:
                for batch_idx, x in pbar:
                    # 1. Move input to device
                    x = x.to(self.device)
                    # 2. Zero the gradients
                    self.optimizer.zero_grad()
                    # 3. Forward pass
                    x_hat, mu, log_var = self(x)
                    # 4. Compute loss
                    loss = self.loss_function(x, x_hat, mu, log_var)
                    # 5. Backward pass
                    loss.backward()
                    # 6. Update weights
                    self.optimizer.step()
                    # Update progress bar
                    try:
                        l = loss.detach().cpu().item()
                    except RuntimeError as e:
                        logger.error(f"Error: {e}")
                        # l = np.nan
                        breakpoint()
                    train_loss += l
                    pbar.set_postfix({"Loss": l})
                logger.info(f"Training Loss: {train_loss / len(train_loader)}")

            if val_loader is None:
                continue
            self.eval()
            val_loss = 0
            for x in val_loader:
                x = x.to(self.device)
                x_hat, mu, log_var = self(x)
                loss = self.loss_function(x, x_hat, mu, log_var)
                val_loss += loss.item()
            logger.info(f"Validation Loss: {val_loss / len(val_loader)}")


if __name__ == "__main__":
    set_seeds()
    config = VAEConfig(hidden_dims=(400, 200), lr=1e-2)
    model = VAE(config)
    model_dir = RESOURCES / "models" / "vae" / "test"

    # Save model
    model.save_model(model_dir)

    # Load model
    m = VAE.load_model(model_dir)
    assert m.config == config

    # Create simple artificial train set of 1 image
    x = torch.rand(1, 784)
    train_loader = DataLoader(x, batch_size=1)

    # Train model
    model.train_loop(epochs=3, train_loader=train_loader)

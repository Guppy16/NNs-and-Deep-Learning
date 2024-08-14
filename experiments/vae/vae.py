from pathlib import Path
from typing import List, TypeVar
import torch
import torch.nn as nn

from tqdm import tqdm
from experiments.base import logger, DEVICE, set_seeds, RESOURCES, TBLogger
from omegaconf import OmegaConf
from dataclasses import dataclass
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader


T = TypeVar("T")


@dataclass
class VAEConfig:
    """Config dataclass for VAE NN"""

    # Model parameters
    hidden_dims: tuple[int, ...]  # e.g. (400, 200), (512, 256, 128)
    input_dim: int = 784
    latent_dim: int = 20

    # device: str = str(DEVICE)

    batch_size: int = 32

    lr: float = 1e-3
    weight_decay: float = 1e-2

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
                    nn.SiLU(),
                )
            )
            in_dim = out_dim

        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(self.config.hidden_dims[-1], self.config.latent_dim)
        self.log_var = nn.Linear(self.config.hidden_dims[-1], self.config.latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

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
                    nn.SiLU(),
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

    def __init__(self, config: VAEConfig, tb_logger: TBLogger):
        super(VAE, self).__init__()
        self.config = config
        self.device = torch.device(DEVICE)

        self.encoder = Encoder(config)
        self.softplus = nn.Softplus()
        self.decoder = Decoder(config)

        self.to(self.device)
        self.optimizer = AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        self.tb_logger = tb_logger
        self.global_step = 0  # For tensorboard logging

    def reparemeterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick to sample z from N(mu, var) distribution.

        z = mu + std * eps

        For more stable training, we use softplus instead of exp for std:
        std = softplus(log_var) = log(1 + exp(log_var))
        Hence, in this case, the encoder learns to predict:
        log_var = log(exp(std) - 1), which isn't exactly the log_var
        but should be more stable
        """
        # std = torch.exp(0.5 * log_var)  # std = sqrt(var)
        std = self.softplus(log_var)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def forward(self, x: Tensor):
        mu, log_var = self.encoder(x)

        # mu, log_var -> N(mu, Σ) distribution
        scale = self.softplus(log_var)
        scale_matrix = torch.diag_embed(scale)
        z_dist = torch.distributions.MultivariateNormal(mu, scale_matrix)

        # Sample z from distribution
        z: Tensor = z_dist.rsample()  # Sampled using reparameterization trick

        x_hat = self.decoder(z)

        return x_hat, z, z_dist

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        z: Tensor,
        z_dist: torch.distributions.Distribution,
    ):
        """Return loss = reconstruction_loss + KLD
        x: batch x img_dim, e.g. 32 x 768
        z: batch x latent_dim, e.g. 32 x 20
        """
        mse_loss = (
            0.5 * nn.functional.mse_loss(x_hat, x, reduction="none").sum(-1).mean()
        )
        bce_loss = (
            nn.functional.binary_cross_entropy(x_hat, x, reduction="none")
            .sum(-1)
            .mean()
        )

        std_normal_dist = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=DEVICE),
            scale_tril=torch.eye(z.shape[-1], device=DEVICE)  # L x L
            .unsqueeze(0)  # 1 x L x L
            .expand(z.shape[0], -1, -1),  # B x L x L
        )
        KLD = torch.distributions.kl_divergence(z_dist, std_normal_dist).mean()

        self.tb_logger.add_scalar("Loss/BCE", bce_loss, self.global_step)
        self.tb_logger.add_scalar("Loss/MSE", mse_loss, self.global_step)
        self.tb_logger.add_scalar("Loss/KLD", KLD, self.global_step)

        loss = bce_loss + KLD
        return loss

    def save_model(self, model_dir: Path) -> tuple[Path, Path]:
        """Save model and config to model_path and return file locations"""

        model_dir.mkdir(parents=True, exist_ok=True)

        # save model
        model_path = model_dir / VAE.MODEL_FILENAME
        torch.save(self.state_dict(), model_path)
        logger.info("Model saved to %s ", model_path)

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

        model = VAE(config, tb_logger=None)
        # load state dict
        model_path = model_dir / VAE.MODEL_FILENAME
        state_dict = torch.load(model_path, map_location=DEVICE)
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
                    self.tb_logger.add_scalar("Loss/train", l, self.global_step)
                    pbar.set_postfix({"Loss": l})
                    self.global_step += 1

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
            self.tb_logger.add_scalar(
                "Loss/val", val_loss / len(val_loader), self.global_step
            )


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

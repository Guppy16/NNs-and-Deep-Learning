"""Script for conditional flow matching on simple moons dataset"""

import itertools
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from sklearn.datasets import make_moons
from torch import Tensor, nn
from tqdm import tqdm

from experiments.base import DEVICE, RESOURCES, SEED, BaseConfig, BaseModel, TBLogger


@dataclass
class FlowConfig(BaseConfig):
    dim: int = 2
    hidden: int = 64


class Flow(BaseModel):
    def __init__(self, config: FlowConfig, **kwargs):
        super().__init__()
        self.config = config

        dim = self.config.dim
        h = self.config.hidden

        self.net = nn.Sequential(
            # time + dim -> hidden
            nn.Linear(1 + dim, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim),  # output is dim
        )

    def forward(
        self, x_t: Float[Tensor, "b d"], t: Float[Tensor, "b"]
    ) -> Float[Tensor, "b d"]:
        x = torch.cat((t, x_t), -1)
        return self.net(x)

    @torch.inference_mode()
    def step(
        self,
        x_t: Float[Tensor, "b d"],
        t_start: Float[Tensor, "1"],
        t_end: Float[Tensor, "1"],
    ) -> Float[Tensor, "b d"]:
        # repeat t_start to fill batch dimension
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)

        # Midpoint ODE solver
        x_t1 = x_t
        x_t2 = self.forward(x_t1, t_start)

        T = t_end - t_start

        dxdt = self.forward(x_t1 + x_t2 * T / 2, t_start + T / 2)

        dx = T * dxdt

        return x_t1 + dx

    @torch.inference_mode()
    def sample(self, batch=300, n_steps=8):
        x = torch.randn(batch, self.config.dim)
        time_steps = torch.linspace(0, 1.0, n_steps + 1)

        samples = torch.empty((n_steps + 1, *x.shape), device=x.device, dtype=x.dtype)
        samples[0] = x

        for i, (t1, t2) in enumerate(itertools.pairwise(time_steps)):
            x = self.step(x, t1, t2)  # update x
            samples[i + 1] = x

        return samples, time_steps


@dataclass
class FlowTrainerConfig(BaseConfig):
    epochs: int = 10000
    batch_size: int = 256
    data_noise: float = 0.05

    learning_rate: float = 1e-2


class FlowTrainer:
    def __init__(self, config: FlowTrainerConfig, model_config: FlowConfig):
        self.config = config
        self.device = DEVICE

        self.model = Flow(model_config)
        self.model.count_parameters()
        self.model = self.model.to(self.device)

        self.optimiser = torch.optim.Adam(
            self.model.parameters(), self.config.learning_rate
        )
        self.loss = nn.MSELoss()

        self.model_dir = (
            RESOURCES
            / "models"
            / "flow"
            / (datetime.now().strftime("%Y%m%d-%H%M%S") + "")
        )
        # self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_config(self.model_dir)

        self.tb_logger = TBLogger(self.model_dir / "logs")

    def training_step(self, x_0, x_1, t):
        """Training step with MSE Loss"""

        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        self.optimiser.zero_grad()

        dxdt = self.model(x_t=x_t, t=t)

        loss = self.loss(dxdt, dx_t)
        loss.backward()
        self.optimiser.step()

        loss_value = loss.cpu().detach().item()
        return loss_value

    def train(self):
        # rng = torch.Generator().manual_seed(SEED)
        rng = np.random.RandomState(SEED)
        self.model.train()

        with tqdm(
            range(self.config.epochs),
            total=self.config.epochs,
            unit="batch",
        ) as pbar:
            for i in pbar:
                x_1 = Tensor(
                    make_moons(
                        self.config.batch_size,
                        noise=self.config.data_noise,
                        random_state=rng,
                    )[0]
                )
                x_0 = torch.randn_like(x_1)
                t = torch.rand(len(x_1), 1)  # i.e. 1x1

                loss_value = self.training_step(x_0, x_1, t)
                self.tb_logger.add_scalar("loss/train", loss_value, i)
                pbar.set_postfix({"Loss": loss_value})

    def plot_samples(self):

        n_steps = 8
        samples, time_steps = self.model.sample(n_steps=n_steps)

        fig, axs = plt.subplots(
            1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True
        )

        for step, (x, t) in enumerate(zip(samples, time_steps)):
            axs[step].scatter(x[:, 0], x[:, 1], s=10)
            axs[step].set_title(f"t={t:.2f}")

        axs[0].set_xlim(-3, 3)
        axs[0].set_ylim(-3, 3)

        plt.tight_layout()
        plt.savefig(self.model_dir / "samples.png")


if __name__ == "__main__":
    model_config = FlowConfig()
    trainer_config = FlowTrainerConfig()

    trainer = FlowTrainer(trainer_config, model_config)
    trainer.train()
    trainer.model.save_model(trainer.model_dir / "model")
    trainer.plot_samples()

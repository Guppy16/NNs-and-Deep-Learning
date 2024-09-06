from experiments.vae.vae import (
    VAE,
    VAEConfig,
    set_seeds,
    RESOURCES,
    TBLogger,
    logger,
    callback_save_zdist,
)
from experiments.data import MNISTDataset, train_val_dataset_split
from torch.utils.data import DataLoader
import torch
from datetime import datetime
from torchvision.utils import save_image
from pathlib import Path

set_seeds()


model_dir = (
    RESOURCES
    / "models"
    / "vae"
    / (datetime.now().strftime("%Y%m%d-%H%M%S") + "l2_e1+b10")
)
# model_dir = RESOURCES / "models" / "vae" / "test"
tb_logger = TBLogger(model_dir / "logs")

config = VAEConfig(
    hidden_dims=(256, 128),
    latent_dim=2,
    lr=1e-3,
    batch_size=32,
    beta=10,
)
model = VAE(config, tb_logger)

train_set, test_set = MNISTDataset()
train_set, val_set = train_val_dataset_split(train_set, val_size=10000)

train_loader = DataLoader(
    train_set,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
)
val_loader = DataLoader(
    val_set,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=4,
)
test_loader = DataLoader(
    test_set,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=4,
)


@torch.no_grad()
def on_nbatch_end(
    model: VAE,
    epoch_idx: int,
    batch_idx: int,
    **_,
):
    if batch_idx % 4 != 0:
        return

    labels_list = []
    z_mu_list = []
    z_cov_list = []

    z_dist: torch.distributions.MultivariateNormal
    for x, labels in test_loader:
        x = x.to(model.device)
        x_hat, z, z_dist = model(x)
        labels_list.extend(labels.detach().cpu())
        z_mu_list.extend(z_dist.loc.detach().cpu())
        z_cov_list.extend(z_dist.covariance_matrix.detach().cpu())

    fpath = Path(model.tb_logger.log_dir).parent / "z_dist"
    fpath.mkdir(parents=True, exist_ok=True)

    fname = f"z_mu_{epoch_idx:03d}_{batch_idx:05d}.pt"
    torch.save(torch.stack(z_mu_list), fpath / fname)

    fname = f"z_cov_{epoch_idx:03d}_{batch_idx:05d}.pt"
    torch.save(torch.stack(z_cov_list), fpath / fname)

    z_cov = torch.mean(z_cov_list, dim=0)
    

    fname = f"labels_{epoch_idx:03d}_{batch_idx:05d}.pt"
    torch.save(labels_list, fpath / fname)


# Print number of parameters
logger.info("Number of parameters: %s", model.repr_parameters())

try:
    model.train_loop(
        epochs=1,
        train_loader=train_loader,
        val_loader=val_loader,
        on_batch_end=on_nbatch_end,
    )
finally:
    model.save_model(model_dir)

# Run forward pass on validation set and save images
model.eval()
val_dir = model_dir / "val"
val_dir.mkdir(parents=True, exist_ok=True)
for i, (x, labels) in enumerate(val_loader):
    x = x.to(model.device)
    x_hat, _, _ = model(x)
    x_hat = x_hat.view(-1, 1, 28, 28)
    save_image(x_hat, val_dir / f"{i}.png")
    if i == 10:
        break
print("Done")

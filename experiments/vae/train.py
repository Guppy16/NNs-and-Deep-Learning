from experiments.vae.vae import VAE, VAEConfig, set_seeds, RESOURCES
from experiments.data import MNISTDataset, train_val_dataset_split
from torch.utils.data import DataLoader
import torch

set_seeds()

config = VAEConfig(hidden_dims=(400, 200), lr=2e-3)
model = VAE(config)
model_dir = RESOURCES / "models" / "vae" / "0806"

train_set, test_set = MNISTDataset()
train_set, val_set = train_val_dataset_split(train_set, val_size=10000)

train_loader = DataLoader(
    train_set,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=lambda x: torch.stack([i[0] for i in x]),
)
val_loader = DataLoader(
    val_set,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=lambda x: torch.stack([i[0] for i in x]),
)


model.train_loop(epochs=10, train_loader=train_loader, val_loader=val_loader)

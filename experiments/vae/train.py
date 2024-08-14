from experiments.vae.vae import VAE, VAEConfig, set_seeds, RESOURCES, TBLogger
from experiments.data import MNISTDataset, train_val_dataset_split
from torch.utils.data import DataLoader
import torch
from datetime import datetime
from torchvision.utils import save_image

set_seeds()


model_dir = (
    RESOURCES
    / "models"
    / "vae"
    / (datetime.now().strftime("%Y%m%d-%H%M%S") + "latent3")
)
# model_dir = RESOURCES / "models" / "vae" / "test"
tb_logger = TBLogger(model_dir / "logs")

config = VAEConfig(hidden_dims=(400, 200), latent_dim=3, lr=1e-3, batch_size=32)
model = VAE(config, tb_logger)

train_set, test_set = MNISTDataset()
train_set, val_set = train_val_dataset_split(train_set, val_size=10000)

train_loader = DataLoader(
    train_set,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=lambda x: torch.stack([i[0] for i in x]),
    num_workers=4,
)
val_loader = DataLoader(
    val_set,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=lambda x: torch.stack([i[0] for i in x]),
    num_workers=4,
)

try:
    model.train_loop(epochs=10, train_loader=train_loader, val_loader=val_loader)
finally:
    model.save_model(model_dir)

# Run forward pass on validation set and save images
model.eval()
val_dir = model_dir / "val"
val_dir.mkdir(parents=True, exist_ok=True)
for i, x in enumerate(val_loader):
    x = x.to(model.device)
    x_hat, _, _ = model(x)
    x_hat = x_hat.view(-1, 1, 28, 28)
    save_image(x_hat, val_dir / f"{i}.png")
    if i == 10:
        break
print("Done")

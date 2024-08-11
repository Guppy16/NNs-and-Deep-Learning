from experiments.vae.vae import VAE, VAEConfig, set_seeds, RESOURCES, TBLogger
from experiments.data import MNISTDataset, train_val_dataset_split
from torch.utils.data import DataLoader
import torch
from datetime import datetime

set_seeds()


model_dir = RESOURCES / "models" / "vae" / datetime.now().strftime("%Y%m%d-%H%M%S")
tb_logger = TBLogger(model_dir / "logs")

config = VAEConfig(hidden_dims=(400, 200), lr=5e-4)
model = VAE(config, tb_logger)

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


model.train_loop(epochs=1000, train_loader=train_loader, val_loader=val_loader)

model.save_model(model_dir)

# Run forward pass on validation set and save images
model.eval()
for i, x in enumerate(val_loader):
    x = x.to(model.device)
    x_hat, _, _ = model(x)
    x_hat = x_hat.view(-1, 1, 28, 28)
    save_image(x_hat, model_dir / "val" / f"{i}.png")
    if i == 10:
        break

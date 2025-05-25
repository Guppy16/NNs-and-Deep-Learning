# Framework for Experiments

This documents the some of the best practices I have learnt for programming NNs.

## Pytorch training loop

```python
def train_loop(self, ...):
  for epoch in range(epochs):
    for batch in dataloader:
      self.train()
      # 1. Move data to device
      batch = batch.to(device) 
      # 2. Zero gradients
      self.optimizer.zero_grad()
      # 3. Forward pass
      output = self.model(batch)
      # 4. Compute loss
      loss = self.loss_function(output, batch)
      # 5. Backward pass
      loss.backward()
      # 6. Update weights
      self.optimizer.step()
```

## omegaconf + dataclass

configuring NNs and storing (hyper)parameters

```python
@dataclass
class ModelConfig:
  # model parameters
  hidden_dim: int = 128
  # training parameters
  lr: float = 1e-3
  batch_size: int = 32

  def save_config(self, path: Path):
    OmegaConf.save(OmegaConf.to_yaml(self), path)

  @staticmethod
  def load_config(path: Path) -> "ModelConfig":
    conf = OmegaConf.load(path)
    return ModelConfig(**conf)
```

## Tensorboard

for logging: see [`base.py`](https://github.com/Guppy16/NNs-and-Deep-Learning/blob/master/experiments/base.py)

A list of potential avenues to explore:

- pytorch-lightning
- hydra
- fire

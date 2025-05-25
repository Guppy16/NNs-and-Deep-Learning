# NNs and Deep Learning

This repo contains a portfolio of experiments and notes on NNs and DL. The aim is twofold:

- use visualisations to understand the latent space when a NN is training
- keep up to date with NN programming practices

View this repo in website format: [guppy16.github.io/NNs-and-Deep-Learning](https://guppy16.github.io/NNs-and-Deep-Learning/)

## Visualisations

### Visualising the Latent Space of a beta-VAE

> The KL loss in a VAE encourages the encoder to map inputs to latents that are close to the prior distribution $\mathcal{N}(0, I)$. We can visualise this distribution for each class in the MNIST dataset by approximating it as a _Mixture of Gaussians_.

<div>
<div style="justify-content: space-between;">
  <!-- First Image Block -->
  <figure>
    <p align="center">Standard VAE (β = 1)<br></p>
    <p align="center">
    <img src="experiments/vae/vis/class_latents_20240830-194511l2_e1.gif" alt="Standard VAE" width="300">
      </p>
    <figcaption><em>The class distributions converge to very different distributions. Digit 0 has a larger spread.</em></figcaption>
  </figure>

  <!-- Second Image Block -->
  <figure style="text-align: center; margin-left: 20px;">
    <p align="center">β = 10<br></p>
    <p align="center">
    <img src="experiments/vae/vis/class_latents_20240830-221914l2_e1+b10.gif" alt="Image 2" width="300">
    </p>
    <figcaption><em>
    The distribution shapes are more similar, but they still try to converge to different locations.
    </em></figcaption>
  </figure>
</div>
<p align="left">
  <br>
<em>
Both VAEs have the same architecture with a 2D latent space, and were trained for a single epoch. In both cases, the model learns to try and separate the the location of the class distributions, however there is significant overlap between the numbers 4 and 9, which is to be expected. The shapes of the distributions are very similar in the beta VAE, which is due the stronger KL loss.
</em>
</p>
</div>

### Separable Latent Space in Classification

> In classification tasks, a NN learns weights so that it is able to create simple decision boundaries to separate classes in the latent space.

<figure>
  <p align="center">
    <img src="./experiments/classifier/latent-space/latent_space.gif"  width="250" alt="weight masks" >
  </p>
  <figcaption><em>Visualising hidden layer with 3 nodes - each has its own axis (NN layers: {784,10,3,10}). As epoch increases, the learnt weights push each digit class to a corner. Unsurprisingly, digits 4 and 9 have significant overlap! See <a href="./experiments/classifier/latent-space/">this folder</a> for implementation.</em></figcaption>
</figure>

### Linear Transformation as a Mask in Classification

> When there is no non-linearity in the NN, the weights are equivalent to a single linear transformation. In the case of classification, intuitively, we are applying a mask on the input.

<figure>
  <p align="center">
    <img src="./resources/figures/digit_weights_mse.gif"  width="250" alt="weight masks">
  </p>
  <figcaption><em>Weights learnt for each digit in a NN with <b>no</b> hidden layer. This is equivalent to applying a mask / linear transformation. See <a href="./experiments/classifier/chap1-no_hidden_layer-MSE_loss.ipynb">this notebook</a> for implementation.</em></figcaption>
</figure>

## Organisation

```
README.md
experiments/  - NN PyTorch class, training experiments
notes/        - markdown notes for experiments and theory
resources/    - store dataset, model and figures
```

## Notes

The [`notes/`](<notes/>) folder contains markdown notes on the relevant NN theory required for the experiments. It also contains notes and exercises from the book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html).

For each chapter, I have written some notes and answers to most exercises / problems:

- [1 Neural Network Intro](<notes/1-NNs_Intro.md>)
- [2 Backpropagation](<notes/2-Backpropagation.md>)
- [3 Improving Learning](<notes/3-Improving_Learning.md>)

This is a WIP; I have yet to do the later chapters.
I also aim to cover the following topics:

- Notes on Activation Functions
  - Swish, softplus (for VAE to predict variance)
- Regularisation: L1, L2, Dropout, [Continual Backprop](https://www.nature.com/articles/s41586-024-07711-7)
- Grid search over batch-size, lr using hydra

## Framework for Experiments

This documents the best practices I have learnt for programming NNs.

### Pytorch training loop

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

### omegaconf + dataclass

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

### Tensorboard

for logging: see [`base.py`](https://github.com/Guppy16/NNs-and-Deep-Learning/blob/master/experiments/base.py)

A list of potential avenues to explore:

- pytorch-lightning
- hydra
- fire


### Quarto

[Quarto](https://quarto.org/) was used to generate the website for this repo.
Some useful commands:

```bash

# Render the website
quarto render
# Serve the website locally
quarto preview
# Publish the website to GitHub Pages
quarto publish gh-pages
```
# VAEs

Now we look at exploring the latent space of a VAE based on known classes.
The MNIST dataset contains class (digit) labels for each image and we can use this to explore how the latent space is organised for each class.
Our encoder encodes a _single_ image, $x$ into a latent space $z$ as $q(z|x) \sim \mathcal{N}(\mu_x, \Lambda_x)$. So the latent distribution for a class $C$ can be written as:

$$
\begin{align*}
p(z|C)
&= \int p(z|x) p(x|C) dx &&\text{[Marginalisation]} \\
&\approx \frac{1}{|C|} \sum_{x \in C} p(z|x), \quad x \sim p(x|C) &&\text{[Monte Carlo Estimate]}\\
&= \frac{1}{|C|} \sum_{x \in C} \mathcal{N}(z;\ \mu_x, \Lambda_x) &&\text{[VAE Encoder]}
\end{align*}
$$

Note that this is a mixture of Gaussians, where each Gaussian is the latent distribution for a single image. (This is a sum of _distributions_, which should not be confused with summing Gaussian _variables_, the latter of which would be a Gaussian with a larger variance).

We note that our prior on the latent space is $z \sim \mathcal{N}(0, I)$ and we try to enforce this with the KL term in the loss. However, perhaps counterintuitively, this does _not_ mean that the $z|C$ must also be $\mathcal{N}(0,1)$. In fact, we would expect the distributions to be rather different so that the decoder can learn to generate the correct digit from the latent space, which is emposed by the reconstruction loss.

1 7

2 3 5

0 4 6 9 8

> Exercise: Calculate the mean and covariance of the latent space for each digit class.

The mean is given as follows

$$
\begin{align*}
\mu_C &= \int z p(z|C) dz \\
&= \int z \frac{1}{|C|} \sum_{x \in C} \mathcal{N}(z;\ \mu_x, \Lambda_x) dz \\
&= \frac{1}{|C|} \sum_{x \in C} \int z \mathcal{N}(z;\ \mu_x, \Lambda_x) dz \\
&= \frac{1}{|C|} \sum_{x \in C} \mu_x,
\end{align*}
$$

where the last line follows from the definition of the mean of a Gaussian.
Intuitively, this is the average of the latent space for each image in the class.

To find the covariance, we use the definition $\Sigma_C = \mathbb{E}[zz^T] - \mathbb{E}[z]^2$.

$$
\begin{align*}
\mathbb{E}[zz^T] &= \int zz^T p(z|C) dz \\
&= \int zz^T \frac{1}{|C|} \sum_{x \in C} \mathcal{N}(z;\ \mu_x, \Lambda_x) dz \\
&= \frac{1}{|C|} \sum_{x \in C} \int zz^T \mathcal{N}(z;\ \mu_x, \Lambda_x) dz \\
&= \frac{1}{|C|} \sum_{x \in C} \ \Lambda_x + \mu_x \mu_x^T,
\end{align*}
$$

Thus, the covariance is

$$
\begin{align*}
\Sigma_C &= \mathbb{E}[zz^T] - \mathbb{E}[z]^2 \\
&= \frac{1}{|C|} \sum_{x \in C} \ \Lambda_x + \mu_x \mu_x^T - \left(\frac{1}{|C|} \sum_{x \in C} \mu_x\right)^2.
\end{align*}
$$

## Experiment Ideas

Visualise training of latent space in 2D / 3D - does this have the same representation as the classification exercise?

- There are three phases to the KL loss (which is very apparent for `latent_dim=8`). Does this reflect in the latent space?!:
    1. KLD starts high, due to random initialisation of the model.
    2. KLD drops almost to 0, as the model learns to encode as a Gaussian very easily
    3. KLD rises sharply to compensate for the high MSE reconstruction loss.
    4. KLD increases slowly to match the MSE loss.

- Use a CNN / UNet as the encoder / decoder

### Visualising higher dimensional latent spaces

So far, we have been able to visualise the latent space of a VAE for 2D and 3D latent spaces.
We require a method to do this for higher dimensional latent spaces. Here are some options

- Potential methods to visulaise high dim space
  - For each dim, use violin / box plot to show distribution
    - If comparing two different runs, then use a violin plot with a violin for each run
  - Perhaps can wrap these around a polar plot

- We can plot e.g. the volume and direction of the latents. For higher dims, we could graph the volumne and directions

- Find path that traverses all digits in latent space

### Beta VAEs

We would also want to visualise the effect of beta more clearly

- Perhaps use plotly with a slider?
- Is there a method to determine the optimal beta?

- show how sliders affect the reconstruction: vary the value of each latent from -3 to 3 (3 sigmas)

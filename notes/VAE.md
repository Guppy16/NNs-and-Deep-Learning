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

## Sampling from the latent space

In this section, we aim to generate samples $x \sim p(x | \mathcal{D})$ within the distribution of our datasets $\mathcal{D}$. The methods involve various methods of sampling from the latent space and passing through the decoder.

1. Sample from the Prior (Naive)

$$
\begin{align*}
  z \sim \mathcal{N}(0, I) \\
\end{align*}
$$

2. Sample from the dataset and then the latent space (Monte Carlo)

$$
\begin{align*}
  x &\sim p_d(x) \\
  z &\sim \mathcal{N}(\mu_x, \Lambda_x)
\end{align*}
$$

3. pCN sampling (MCMC)

This is a bit more involved.

We only use the decoder

Metropolis-Hastings algorithm:

1. Sample $z_0 \sim \mathcal{N}(0, I)$
2. For $i = 1, \ldots, N$:
    1. Sample from proposal $z' \sim \mathcal{P}(z_i)$
    2. Calculate the acceptance threshold $\bar\alpha$
    3. Sample $u \sim \mathcal{U}(0, 1)$
    4. If $u < \bar\alpha$, then $z_{i+1} = z'$, else $z_{i+1} = z_i$

Note the following

* the acceptance probability is $\alpha = \min(1, \bar\alpha)$, but we do not need to calculate this explicitly as we can just compare $u$ to $\bar\alpha$
* in step 2.4 we may _reject_ a sample, but this does not mean that the chain is stuck at generating the same image - this is because the decoder is in fact sampling from $p_\theta(x|z) = \mathcal{N}(\text{Decode}(x), I)$. So the decoder will generate a different image for the same latent space.

For pre-conditioned Crank-Nicolson (pCN) sampling, the proposal, $\mathcal{P}(z_i \to z')$, is given by

$$
z' = \sqrt{1 - \beta^2} \ z_i + \beta \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
$$

For pCN, the acceptance threshold simplifies as follows:

$$
\bar\alpha
= \frac{p(z'|\mathcal{D}) \mathcal{P}(z_i \to z')}{p(z_i|\mathcal{D}) \mathcal{P}(z' \to z_i)}  
= \frac{p(\mathcal{D}|z')}{p(\mathcal{D}|z_i)}
$$

This simplification works out due to the construction of the pCN proposal, which we prove later. We note that this simplification is a remarkable result, as it means that we only require the likelihood, $p(\mathcal{D}|z)$, to calculate the acceptance threshold, and do not deal with the prior or proposal distributions, which may otherwise result in a low acceptance rate.

Under our VAE model, the likelihood is given by our decoder:

$$
\begin{align*}
  p(\mathcal{D}|z)
  &= \prod_i p_\theta(x^i|z)  \\
  &\propto \exp\left(-\frac{1}{2} \sum_i \left\|x^i - \text{Decode}(z)\right\|_2^2\right)
  .
\end{align*}
$$

> Exercise: Prove the pCN acceptance threshold simplification. (We ignore the measure theoretic details).

$$
\begin{align*}
  
\end{align*}
$$

## Experiment Ideas

Visualise training of latent space in 2D / 3D - does this have the same representation as the classification exercise?

* There are three phases to the KL loss (which is very apparent for `latent_dim=8`). Does this reflect in the latent space?!:
    1. KLD starts high, due to random initialisation of the model.
    2. KLD drops almost to 0, as the model learns to encode as a Gaussian very easily
    3. KLD rises sharply to compensate for the high MSE reconstruction loss.
    4. KLD increases slowly to match the MSE loss.

* Use a CNN / UNet as the encoder / decoder

### Visualising higher dimensional latent spaces

So far, we have been able to visualise the latent space of a VAE for 2D and 3D latent spaces.
We require a method to do this for higher dimensional latent spaces. Here are some options

* Potential methods to visulaise high dim space
  * For each dim, use violin / box plot to show distribution
    * If comparing two different runs, then use a violin plot with a violin for each run
  * Perhaps can wrap these around a polar plot

* We can plot e.g. the volume and direction of the latents. For higher dims, we could graph the volumne and directions

* Find path that traverses all digits in latent space

### Beta VAEs

We would also want to visualise the effect of beta more clearly

* Perhaps use plotly with a slider?
* Is there a method to determine the optimal beta?

* show how sliders affect the reconstruction: vary the value of each latent from -3 to 3 (3 sigmas)

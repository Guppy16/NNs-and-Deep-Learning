# How the backpropagation algorithm works

This markdown file contains my notes on Chapter 2 of the book [Neural Networks and Deep Learning]([url](http://neuralnetworksanddeeplearning.com/chap2.html)).
It is essentially a bank of equation derivations:
- Mean Square Error Cost function
- 4 Fundamental Equations
  - Error due to output layer weights
  - Recursive equation for layer error
  - Error due to bias
  - Error due to weight in any layer
- Backpropagation algorithm

The backpropagation algorithm was introduced in the 70s, but was better understood in a famous [1986 paper by Rumelhart, Hinton, Williams](http://www.nature.com/nature/journal/v323/n6088/pdf/323533a0.pdf)

## A fast matrix-based approach to computing the output from a neural network

- $w_{jk}^l$ refers to the $k$-th neuron in the $l-1$-th layer to the $j$-th neuron on the $l$-th layer.
- $a_j^l$ is the activation of the $j$-th neuron in the $l$-th layer.
- $b_j^l$ is the bias in the $j$-th neuron in the $l$-th layer

$$
a_j^l = \sigma \left( \sum_k w_{jk}^l a_k^{l-1} + b_j^l \right)
$$

In matrix form, we have:

$$
\begin{align*}
  z^l &:= w^l \cdot a^{l-1} + b^l \\
  a &= \sigma \left ( z^l \right)
\end{align*}
$$

- $z^l$ is the _weighted input_ to the neurons in the layer $l$
- $w^l$ as the weight matrix defining the weights connecting the layer $l-1$ to $l$. i.e. the $w_{jk}^l$ refers to the $j$-th row and $k$-th column.

## Assumption we need about the cost function

1. The cost function $C_x$ wrt to a training example $x$ is independent of all other training examples: e.g. $C = \frac{1}{n} \sum_x C_x$
   - This is necessary so that $\partial C/ \partial w$ can be calculated for each training example and aggregated in e.g. SGD
2. The cost function $C$ can be written as a continuous function of the NN output (i.e the activations in last layer). e.g. $C(a^L) = \lVert y - a^L \rVert^2$
   - Note: $y$ is not a variable in the cost function $C$ in the sense that $y$ is a fixed parameter used to define the function.

## Hadamard product

Element-wise product of two matrices of the same dimension commonly referred to as _Hadamard product_ or _Schur product_:

$$
\begin{bmatrix}
  1 \\ 2
\end{bmatrix}
\odot
\begin{bmatrix}
  3 \\ 4
\end{bmatrix}
  =
\begin{bmatrix}
  1 \times 3 \\ 2 \times 4
\end{bmatrix}
=
\begin{bmatrix}
  3 \\ 8  
\end{bmatrix}
$$

## Fundamental equations behind backpropagation

The error in the cost wrt the _weighted input_ in the $j$-th neuron in the $l$-th layer can be interpreted as $\delta_j^L := \frac{\partial C}{\partial z_j^l}$. Intuitively, if the error is small, then we have reached a local minima.

Note: For this section, we will use [denominator-layout notation](https://en.wikipedia.org/wiki/Matrix_calculus#Denominator-layout_notation), which is the convention used by the book. However libraries such as `pytorch` and `numpy` use numerator-layout convention:

i.e.

- $z$ is a column vector
- $\frac{\partial C}{\partial z}$ is a column vector with elements $\frac{\partial C}{\partial z_i}$
- $\frac{\partial a}{\partial z}$ is a matrix with the elements $J_{ij} = \frac{\partial a_j}{\partial z_i}$

### BP1. The error in cost function wrt the weights in the output layer

$$
\begin{align*}
  \vec{\delta}^L &= \frac{\partial C(\vec{a}^L)}{\partial \vec{z}^L} \\
   &= \left[\frac{\partial \vec{a}^L}{\partial \vec{z}^L}\right] \frac{\partial C(\vec{a}^L)}{\partial \vec{a}^L}  \\
  &= \Sigma'(z^L) \nabla_{a^L} C \\
\end{align*}
$$

where $\Sigma'$ is a diagonal matrix because $\frac{\partial a_j^L}{\partial z_i^L} = 0$ for $i \ne j$ (i.e. $a_i^L$ is does not depend on $z_j^L$ for $i \ne j$). Subbing in for our equations, this can be written in vectorised form:

$$
\begin{align*}
   \vec{\delta}^L &= \left[ \frac{\partial}{\partial \vec{a}^L} \frac{1}{2} \lVert \vec{y} - \vec{a}^L\rVert_2^2 \right] \text{diag} \left[\frac{\partial}{\partial \vec{z}^L} \sigma(\vec{z}_L)\right] \\
  &= (a^L - y) \odot \sigma'(z^L) \\
\end{align*}
$$

### BP2. Recursive equation for layer error

Equation for the error $\delta^l$ in terms of the error in the next layer $\delta^{l+1}$

$$
\begin{align*}
  \vec{\delta}^l &= \frac{\partial C}{\partial \vec{z}^l} \\
  &= \left[ \frac{\partial z^{l+1}}{\partial z^l} \right] \frac{\partial C}{\partial z^{l+1}}  \\
  &= \left[ \frac{\partial z^{l+1}}{\partial z^l} \right] \delta^{l+1} \\
  \text{Consider:} \\
  \frac{\partial}{\partial z^l} \{ z^{l+1} \} &= \frac{\partial}{\partial z^l} \{ w^{l+1} \sigma(z^l) + b^{l+1} \} \\
  \implies \frac{\partial z^{l+1}}{\partial z^l} &= \left[ \frac{\partial \vec{\sigma}}{\partial \vec{z^l}} \right] \frac{\partial}{\partial \vec{\sigma}} w^{l+1} \vec{\sigma}  \\
  &= \Sigma'(z^l) \ (w^{l+1})^T \\
  \text{Subbing in:} \\
  \delta^l &= \Sigma'(z^l) \ (w^{l+1})^T \ \delta^{l+1} \\
  &= \sigma'(z^l) \odot (w^{l+1})^T \delta^{l+1}
\end{align*}
$$

### BP3. Derivative of the cost wrt the bias

$$
\begin{align*}
  \frac{\partial C}{\partial \vec{b}^l}
  &= \frac{\partial \vec{z}^l}{\partial \vec{b}^l} \frac{\partial C}{\partial \vec{z}^l} \\
  &= \frac{\partial}{\partial \vec{b}^l} \{w^la^{l-1} + b^l\} \vec{\delta}^l \\
  &= \vec{\delta}^l
\end{align*}
$$

### BP4. Derivative of the cost wrt any weight in the network

This is better approached using index notation (because $\partial z / \partial w$ is a 3rd order tensor!)

$$
\begin{align*}
  \frac{\partial C}{\partial w_{jk}^l}
  &= \frac{\partial \vec{z}^l}{\partial w_{jk}^l} \frac{\partial C}{\partial \vec{z}^l} \\
  &= \frac{\partial}{\partial w_{jk}^l} \{w^la^{l-1} + b^l\} \vec{\delta}^l \\
  &= a_k^{l-1} \vec{\delta}_j^l \\
\end{align*}
$$

Thus:

$$
\begin{align*}
\implies \frac{\partial C}{\partial w^l} &= \delta^l {a^{l-1}}^T \\
&= a_\text{in} \delta_\text{out}
\end{align*}
$$

Some insights;

- $a_\text{in} \approx 0 \implies \partial C / \partial w$  will be small, hence the weights will learn slowly. i.e. low activation neurons learn slowly
- When $a \approx \in \{0,1\} \implies \sigma' \approx 0$, hence the weights and biases in the _final_ layer will learn slowly if the output neuron is _saturated_.
- Similarily if $\sigma'(z^l) \ll (w^{l+1})^T \delta^{l+1}$ , then the weights and biases in previous layers will learn slowly.
- To get around _saturation_, one can use an activation function that has a constant / increasing gradient.

## Backpropagation Algorithm

1. _Input_ $x$ Set the corresponding activation for the input layer $a^{l=1}$ (I believe the input _literally_ just sets the activations in the first layer)
2. _Feedforward_: For each $l=\{1,2,\ldots,L \}$, compute: $z^l$ and $a^l$
3. _Output error_: Compute the output layer error $\delta^L$ (using BP1)
4. _Backpropagate_ the error: For each $l=\{L-1, L=2, \ldots, 2\}$ compute $\delta^l$ (using BP2)
5. _Update_: Calculate the gradients wrt the weights and biases using BP3 and BP4 and update weights and biases. Note that when using SGD, the updates will be done on the average gradient of the _mini-batch_. 

### Exercises

> Backpropagation with a single modified neuron

If the activation function of a _single_ neuron was modified to $f(z) \ne \sigma(z)$, then $\sigma'(z^l)$ would be modified such that the corresponding element would be $f'(z^l_k)$

> Backpropagation with linear neurons

This corresponds to setting $\sigma'(z^l) = \vec{1}$

## In what sense is backpropagation fast

Backpropagation requires two passes through the network: forward and backward, which are both similar in complexity (Note that there are no inverse matrices required to be calculated!).

A naive approach may be to find $\partial C / \partial w_{jk}^l$ for each weight by using a first order apporximation and running the network twice to find $C(w), C(w + \epsilon)$. This scales with the number of parameters, which is incredibly slow. Conversely, backpropagation utilises the chain rule to update all the parameters based of one observation, by utilising the relationship between the weights.

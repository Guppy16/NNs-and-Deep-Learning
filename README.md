# NNs-and-Deep-Learning

This repo tracks my notes and exercises while completing the book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html).

![Initial Layer weights](resources/figures/digit_weights_mse.gif)

> This gif shows the weights learnt for each digit in a NN with no hidden layer


## Organisation

```
README.md
experiments/  - NN PyTorch class, training experiments
notes/        - markdown notes for each chapter
resources/    - store dataset, model and figures
```

The bulk of the Neural Network class is in `experiments/digit_classifier.py`. This has been implemented using PyTorch. 

## Chapter Notes

For each chapter, I have written some notes and answers to most exercises / problems:

- [Chap 1](notes/chap1.md)
- [Chap 2](notes/chap2.md)
- [Chap 3](notes/chap3.md)

This is a WIP; I have yet to do the later chapters.

## Optimising Neural Nets

There are many possible avenues to explore with optimising Neural Nets:

- [x] Loss function: MSE, Cross Entropy
- [x] Grid search for mini-batch and learning rate
- [x] Weight init (automagically done in PyTorch)
- [ ] Learning rate schedule: hold learning rate constant until the _validation_ accuracy worsens, then decrease $\eta$ by 2-10x
- [ ] Regularisation
  - [ ] L1
  - [ ] L2
  - [ ] Dropout
  - [ ] ~~Dataset Augmentation~~
- [ ] Softmax in the output layer (This may be demonstrated in a later chapter)
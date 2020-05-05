## DCGAN：generate fake faces with DCGAN

```
---
title: DCGAN: generate fake faces with DCGAN
author: librahg
date: 2020/5/5
---
```

#### Introduction

GANs are a framework for teaching a DL model to capture the training data’s distribution so we can generate new data from that same distribution. They are made of two distinct models, a *generator* and a *discriminator*. The job of the generator is to spawn ‘fake’ images that look like the training images. The job of the discriminator is to look at an image and output whether or not it is a real training image or a fake image from the generator. 

During training, the generator is constantly trying to outsmart the discriminator by generating better and better fakes, while the discriminator is working to become a better detective and correctly classify the real and fake images. 

The equilibrium of this game is when the generator is generating perfect fakes that look as if they came directly from the training data, and the discriminator is left to always guess at 50% confidence that the generator output is real or fake.

#### Models

Generator: input(vector that encodes randomly),  output(fake data that likes training data).

![dcgan_generator](https://pytorch.org/tutorials/_images/dcgan_generator.png)

Discriminator: Input(training data),  output(single element(0~1))

#### Loss Function

**BCEloss**: $\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]$

```
This is used for measuring the error of a reconstruction in for example
an auto-encoder.
```

#### Training

1. **Train the Discriminator**

   First, we will construct a batch of real samples from the training set, forward pass through Discriminator network, calculate the loss ($log(D(x)$), then calculate the gradients in a backward pass. Secondly, we will construct a batch of fake samples with the current generator, forward pass this batch through DD, calculate the loss $(log(1−D(G(z)))$, and *accumulate* the gradients with a backward pass. Now, with the gradients accumulated from both the all-real and all-fake batches, we call a step of the Discriminator’s optimizer.

2. **Train the Generator**

   Classifying the Generator output from Part 1 with the Discriminator, computing G’s loss *using real labels as GT*, computing G’s gradients in a backward pass, and finally updating G’s parameters with an optimizer step. 

#### Structure

* data: save npy file with losses, including d_losses, g_losses.

* weights: save the weights of all networks that trained ahead, including Generator and Discriminator.
* networks: the building process of all networks.
* dataloader.py: the reading and loading of face datasets.
* experiments.py: including train, generate fake faces and show.

#### Results

![image](https://user-images.githubusercontent.com/34414402/81046858-d2019800-8eeb-11ea-98d2-e3710c767769.png)

<img src="https://user-images.githubusercontent.com/34414402/81046897-e5acfe80-8eeb-11ea-932c-d184e782c8e3.png" alt="image" style="zoom: 25%;" />

<img src="https://user-images.githubusercontent.com/34414402/81046887-e04fb400-8eeb-11ea-8af0-672ff1d00a9f.png" alt="image" style="zoom:40%;" />

#### Reference

[1] https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

[2] https://arxiv.org/pdf/1511.06434.pdf




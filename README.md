# Anime Face Generation with a Variational Autoencoder

This project implements a **Variational Autoencoder (VAE)** from scratch to generate anime-style face images. The goal is to learn the underlying distribution of anime face images and produce new samples that resemble the training data.

## 1. Image Source

For this project, I will use an anime face dataset from publicly available sources such as **Kaggle**. The dataset consists of aligned anime-style face images, which are well-suited for generative modeling.

The goal is to learn the distribution of these images and generate new anime faces that resemble the original dataset.

## 2. Model Architecture

I will implement a **Variational Autoencoder (VAE)** from scratch, without using any pre-trained models or fine-tuning.

The model includes:

- An **encoder** that maps input images to a latent distribution (mean and variance)
- A **decoder** that reconstructs images from sampled latent variables

The training objective consists of:

- **Reconstruction loss** to ensure similarity between input and output
- **KL divergence loss** to regularize the latent space

This project focuses on a clear and correct implementation of the VAE framework as covered in class.

## 3. Extra Criteria

In addition to the baseline implementation, I will explore the following:

- **Latent space exploration**  
  Interpolating between latent vectors to visualize how generated images change smoothly.

- **Hyperparameter tuning**  
  Testing different latent dimensions and model sizes to analyze their impact on performance.

- **Optional visualization**  
  Visualizing generated samples during training.

## 4. Project Goals

- Build a VAE for anime face generation from scratch
- Train the model on an anime face dataset
- Generate new anime-style face images
- Explore the structure of the latent space
- Analyze how hyperparameters affect generation quality

## 5. Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Kaggle dataset

## 6. Expected Outcomes

By the end of this project, I expect to:

- Successfully train a VAE on anime face images
- Generate visually coherent anime faces
- Demonstrate latent space interpolation
- Evaluate how model settings influence performance

## 7. Future Improvements

Possible future extensions include:

- Using a deeper convolutional VAE
- Comparing VAE results with GAN or diffusion models
- Building a simple interface to sample generated faces interactively

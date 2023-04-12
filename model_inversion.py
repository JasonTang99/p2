import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from time import time

from models import Generator_MNIST, Encoder
from data import load_MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gradient_ascent(model, imgs, latent_dim, lr=1e-2, iterations=1000, z_0_mult=1):
    """Perform gradient ascent on an image to find the optimal latent vector
    model: Generator taking (batch_size, latent_dim) -> (batch_size, 1, 28, 28)
    imgs: (batch_size, 1, 28, 28)
    latent_dim: int
    """
    batch_size = imgs.shape[0]
    # Setup latent vector
    latent_vectors = torch.randn(batch_size, latent_dim).to(device)
    latent_vectors.requires_grad = True

    # Sample z_0 and calculate P_Z(z_0)
    z_0 = torch.randn(batch_size, latent_dim).to(device) * z_0_mult
    norm_z_0 = torch.linalg.norm(z_0, dim=1)

    # Setup optimizer and loss
    optimizer = optim.SGD([latent_vectors], lr=lr)
    criterion = nn.MSELoss()

    # Train
    model.eval()
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(latent_vectors)
        loss = criterion(output, imgs)
        loss.backward()
        optimizer.step()

        # Project any latent vector outside of sphere to the surface of the sphere
        with torch.no_grad():
            norm_latent_vectors = torch.linalg.norm(latent_vectors, dim=1)
            scale = norm_latent_vectors / norm_z_0
            scale = torch.where(scale < 1, torch.ones_like(scale), scale)
            latent_vectors = latent_vectors / scale.unsqueeze(1)

        if i % 50 == 0:
            print(f"Iteration: {i} Loss: {loss.item()}")

    return latent_vectors


def run_wgan(gen_fp, train_loader, save_fp="data/wgan_latent_dataset.pt"):
    """Iterates through all images in train_loader and performs 
        gradient ascent on each image
    """
    # Load model
    model = Generator_MNIST().to(device)
    model.load_state_dict(torch.load(gen_fp))

    # Initialize new latent dataset
    latent_dataset = data.TensorDataset(torch.zeros(0, 100).to(device))

    # Get images
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)

        # Perform gradient ascent
        latent_vectors = gradient_ascent(model, imgs, 100, lr=1e-2,
            iterations=200, z_0_mult=1)
        
        # Save latent vectors to torch dataset
        latent_dataset = data.TensorDataset(torch.cat((latent_dataset.tensors[0], latent_vectors), dim=0))

    # Save latent vectors to file
    torch.save(latent_dataset, save_fp)
    print("Saved latent vectors")


def run_autoencoder(enc_fp, train_loader, save_fp="data/autoencoder_latent_dataset.pt"):
    """Iterates through all images in train_loader and encodes each image
    """
    # Load model
    encoder = Encoder(latent_size=100).to(device)
    encoder.load_state_dict(torch.load(enc_fp))

    # Initialize new latent dataset
    latent_dataset = data.TensorDataset(torch.zeros(0, 100).to(device))

    # Get images
    for i, (imgs, _) in enumerate(train_loader):
        imgs = imgs.to(device)

        # Encode images
        latent_vectors = encoder(imgs)
        
        # Save latent vectors to torch dataset
        latent_dataset = data.TensorDataset(torch.cat((latent_dataset.tensors[0], latent_vectors), dim=0))
    # Print size
    print(latent_dataset.tensors[0].shape)

    # Save latent vectors to file
    torch.save(latent_dataset, save_fp)
    print("Saved latent vectors")

if __name__ == "__main__":
    # Load data
    batch_size = 128
    labeling_loader, public_loader, private_loader, test_loader = load_MNIST(batch_size=batch_size)
    print(len(private_loader.dataset))

    # Run Autoencoder
    enc_fp = "runs/autoencoder/enc.pth"
    run_autoencoder(enc_fp, private_loader, "data/autoencoder_latent_dataset.pt")

    exit(0)

    gen_fp = "runs/public_16-12_100_32_1_inf_1e-06_0.4_0.005_0.0001_0.5_64_4_300000_LeakyReLU/netG_100000.pt"

    run_wgan(gen_fp, public_loader, "data/wgan_latent_dataset.pt")

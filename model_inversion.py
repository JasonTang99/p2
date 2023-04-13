import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from time import time

from models import Generator_MNIST, Encoder_Mini, Decoder_Mini, Discriminator_FC, Generator_FC
from data import load_MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.autograd as autograd

import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def projected_gradient_ascent(model, imgs, latent_dim, start_lr=1.0, iterations=100000, z_0_mult=1):
    """Perform projected gradient ascent on an image to find the optimal latent vector
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


    # Setup loss
    criterion = nn.MSELoss()

    # Train
    model.eval()
    prev_loss = float("inf")
    for i in range(iterations):
        latent_vectors.requires_grad = True
        output = model(latent_vectors)
        loss = criterion(output, imgs)

        # Compute gradient w.r.t. latent vector using autograd
        gradients = autograd.grad(
            outputs=loss,
            inputs=latent_vectors,
            grad_outputs=torch.ones_like(loss),
        )[0]

        # Gradient ascent step
        latent_vectors = latent_vectors - start_lr * gradients

        # Project latent vector back onto the sphere
        with torch.no_grad():
            norm_latent_vectors = torch.linalg.norm(latent_vectors, dim=1)
            scale = norm_latent_vectors / norm_z_0
            scale = torch.max(scale, torch.ones_like(scale))
            latent_vectors = latent_vectors / scale.unsqueeze(1)

        if i % 1000 == 0:
            # if loss increases, halve learning rate
            if loss.item() > prev_loss - 1e-6:
                start_lr /= 2
                if start_lr < 1:
                    break
            prev_loss = loss.item()
        

        if i % 10000 == 0:
            print(f"Loss: {loss.item()}")

        # Delete everything to free up memory
        del output, loss, gradients, norm_latent_vectors, scale

    return latent_vectors


def gradient_ascent(model, imgs, latent_dim, start_lr=10, iterations=1000, latent_vectors=None):
    """Perform gradient ascent on an image to find the optimal latent vector
    model: Generator taking (batch_size, latent_dim) -> (batch_size, 1, 28, 28)
    imgs: (batch_size, 1, 28, 28)
    latent_dim: int
    """
    batch_size = imgs.shape[0]
    # Setup latent vector
    if latent_vectors is None:
        latent_vectors = torch.randn(batch_size, latent_dim).to(device)
    latent_vectors.requires_grad = True

    # Setup optimizer and loss
    optimizer = optim.SGD([latent_vectors], lr=start_lr)
    criterion = nn.MSELoss()

    # Train
    model.eval()
    prev_loss = float("inf")
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(latent_vectors)
        loss = criterion(output, imgs)
        loss.backward()
        optimizer.step()

        # if loss increases, stop
        if i % 100 == 0:
            if loss.item() > prev_loss - 1e-6:
                start_lr /= 2
                if start_lr < 1:
                    break
                for param_group in optimizer.param_groups:
                    param_group['lr'] = start_lr
            prev_loss = loss.item().detach()
        
        if i % 10000 == 0:
            print(f"Loss: {loss.item()}")
        
    return latent_vectors

def run_wgan(gen_fp, train_loader, save_fp="data/wgan_latent_dataset.pt"):
    """Iterates through all images in train_loader and performs 
        gradient ascent on each image
    """
    # Load model
    model = Generator_FC(hidden_sizes=[256], nz=100).to(device)
    model.load_state_dict(torch.load(gen_fp))

    # Initialize new latent dataset
    latent_dataset = data.TensorDataset(torch.zeros(0, 100).to(device))

    # Get images
    for imgs, _ in tqdm(train_loader):
        imgs = imgs.to(device)

        # Perform gradient ascent
        latent_vectors = projected_gradient_ascent(
            model, imgs, latent_dim=100, start_lr=200,
            iterations=200000, z_0_mult=1)
        
        # Save latent vectors to torch dataset
        latent_dataset = data.TensorDataset(torch.cat((latent_dataset.tensors[0], latent_vectors), dim=0))

    # Save latent vectors to file
    torch.save(latent_dataset, save_fp)
    print(f"Saved {latent_dataset.tensors[0].shape[0]} latent vectors")


def run_ae_grad(dec_fp, train_loader, save_fp="data/ae_grad_latent_dataset.pt"):
    """Iterates through all images in train_loader and performs 
        gradient ascent on each image
    """
    # Load model
    decoder = Decoder_Mini(latent_size=100).to(device)
    decoder.load_state_dict(torch.load(dec_fp))

    # Initialize new latent dataset
    latent_dataset = data.TensorDataset(torch.zeros(0, 100).to(device))

    # Get images
    for imgs, _ in tqdm(train_loader):
        imgs = imgs.to(device)

        # Perform gradient ascent
        latent_vectors = gradient_ascent(
            decoder, imgs, latent_dim=100, start_lr=200,
            iterations=200000)
        
        # Save latent vectors to torch dataset
        latent_dataset = data.TensorDataset(torch.cat((latent_dataset.tensors[0], latent_vectors), dim=0))

    # Save latent vectors to file
    torch.save(latent_dataset, save_fp)
    print(f"Saved {latent_dataset.tensors[0].shape[0]} latent vectors")


def run_ae(enc_fp, train_loader, save_fp="data/ae_enc_latent_dataset.pt"):
    """Iterates through all images in train_loader and encodes each image
    """
    # Load model
    encoder = Encoder_Mini(latent_size=100).to(device)
    encoder.load_state_dict(torch.load(enc_fp))

    # Initialize new latent dataset
    latent_dataset = data.TensorDataset(torch.zeros(0, 100).to(device))

    # Get images
    for imgs, _ in tqdm(train_loader):
        imgs = imgs.to(device)

        # Encode images
        latent_vectors = encoder(imgs)
        
        # Save latent vectors to torch dataset
        latent_dataset = data.TensorDataset(torch.cat((latent_dataset.tensors[0], latent_vectors), dim=0))

    # Save latent vectors to file
    torch.save(latent_dataset, save_fp)
    print(f"Saved {latent_dataset.tensors[0].shape[0]} latent vectors")

# Export the models used for latent computations
enc_fp = "runs/autoencoder_mini/enc.pth"
dec_fp = "runs/autoencoder_mini/dec.pth"
gen_fp = "runs_gen_fc_3/public_256_100_32_1_inf_1e-06_0.0_0.01_5e-05_0.0_64_3_500000_LeakyReLU_0.0/netG_470000.pt"


if __name__ == "__main__":
    # Load data
    batch_size = 400
    _, _, private_loader, _ = load_MNIST(batch_size=batch_size)

    print(len(private_loader.dataset))

    # Run Autoencoder Encoder if needed
    if not os.path.exists("data/ae_enc_latent_dataset.pt"):
        run_ae(enc_fp, private_loader, "data/ae_enc_latent_dataset.pt")

    # Run Autoencoder Gradient Ascent
    if not os.path.exists("data/ae_grad_latent_dataset.pt"):
        run_ae_grad(dec_fp, private_loader, "data/ae_grad_latent_dataset.pt")

    # Run WGAN
    if not os.path.exists("data/wgan_latent_dataset.pt"):
        run_wgan(gen_fp, private_loader, "data/wgan_latent_dataset.pt")

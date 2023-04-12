import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from time import time

from models import Encoder, Decoder, Encoder_Mini, Decoder_Mini
from data import load_MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")


def train(enc, dec, train_loader, lr=5e-4, epochs=1000, run_fp="runs/autoencoder", verbose=True):
    """Training process"""
    os.makedirs(run_fp, exist_ok=True)

    # Check if anything inside run_fp exists
    if os.path.exists(f"{run_fp}/loss.txt"):
        print(f"{run_fp} already exists. Skipping...")
        return
    
    # Setup optimizer
    optimizer = optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=lr,
        weight_decay=1e-5
    )

    # Setup loss
    criterion = nn.MSELoss()
    
    # Track time
    start_time = time()
    
    enc.train()
    dec.train()
    with open(f"{run_fp}/loss.txt", "a") as f:
        for i in tqdm(range(epochs)):
            for j, (data, _) in enumerate(train_loader):
                # Train with real data
                optimizer.zero_grad()
                output = dec(enc(data))
                # Reshape output and data
                output = output.view(-1, 28*28)
                data = data.view(-1, 28*28)

                loss = criterion(output, data)
                loss.backward()
                optimizer.step()
                
                if verbose and j % 100 == 0:
                    print(f"Epoch {i} Batch {j} Loss {loss.item()}")
                
                f.write(f"{i} {j} {loss.item()}")
            
            if i != 0 and i % 200 == 0:
                # Save model
                torch.save(enc.state_dict(), f"{run_fp}/enc_{i}.pth")
                torch.save(dec.state_dict(), f"{run_fp}/dec_{i}.pth")
    
        # Track time
        end_time = time()
        print(f"Training took {end_time - start_time} seconds")
        f.write(f"Training took {end_time - start_time} seconds")

    # Save model
    torch.save(enc.state_dict(), f"{run_fp}/enc.pth")
    torch.save(dec.state_dict(), f"{run_fp}/dec.pth")

if __name__ == "__main__":
    # Data
    labeling_loader, public_loader, private_loader, test_loader = load_MNIST(batch_size=64)
    train_loader = public_loader

    # Train Mini Models
    enc = Encoder_Mini(latent_size=100)
    dec = Decoder_Mini(latent_size=100)

    # Train
    train(enc, dec, train_loader, lr=5e-4, epochs=1001, run_fp="runs/autoencoder_mini", verbose=True)
    exit(0)

    # Model
    enc = Encoder(latent_size=100)
    dec = Decoder(latent_size=100)

    # Train
    train(enc, dec, train_loader, lr=5e-4, epochs=1001, verbose=True)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from time import time

from models import Encoder, Decoder
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


def train(enc, dec, train_loader, lr=5e-4, verbose=True):
    """Training process"""
    run_fp = "runs/autoencoder"

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
    epochs = 1000
    
    enc.train()
    dec.train()
    with open(f"{run_fp}/loss.txt", "a") as f:
        for i in tqdm(range(epochs)):
            for j, (data, _) in enumerate(train_loader):
                # Train with real data
                optimizer.zero_grad()
                output = dec(enc(data))
                loss = criterion(output, data)
                loss.backward()
                optimizer.step()
                
                if verbose and j % 10 == 0:
                    print(f"Epoch {i} Batch {j} Loss {loss.item()}")
                    f.write(f"{i} {j} {loss.item()}")

if __name__ == "__main__":
    # Data
    labeling_loader, public_loader, private_loader, test_loader = load_MNIST(batch_size=64)
    train_loader = public_loader

    # Model
    enc = Encoder()
    dec = Decoder()

    # Train
    train(enc, dec, train_loader)
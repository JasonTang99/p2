import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
import os
from time import time

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


def train(run_fp, model, optimizer, train_loader, loss_fn, device, epochs=int(5e3), verbose=False):
    """Training process"""

    # Check if anything inside run_fp exists
    if os.path.exists(f"{run_fp}/loss.txt"):
        print(f"{run_fp} already exists. Skipping...")
        return
    
    # Track time
    start_time = time()
    print_mod = 10
    
    model.train()
    with open(f"{run_fp}/loss.txt", "a") as f:
        for epoch in tqdm(range(epochs)):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)

                # Compute regularized loss
                loss = loss_fn(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Print loss
                if verbose:
                    print(f"Epoch {epoch} | Iteration {i} | Loss {loss.item()}")
                if i % print_mod == 0:
                    print(f"{i} {loss.item()}", file=f)

    print(f"Training time: {time() - start_time:.2f} seconds")
    print(f"Training time: {time() - start_time:.2f} seconds", file=f)

    # Save model
    torch.save(model.state_dict(), f"{run_fp}/model.pt")



def main(lr, weight_decay, batch_size, epochs):
    """Train a labeling model using the labeling set
    """
    # Random Seeding
    torch.manual_seed(0)
    np.random.seed(0)

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate Run ID
    run_fp = 'runs/labeling'
    os.makedirs(run_fp, exist_ok=True)

    # Setup ResNet Model
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)
    model.to(device)

    # Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Setup Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Setup MNIST dataset using load_MNIST
    labeling_loader, public_loader, private_loader, test_loader = load_MNIST(batch_size)
    train_loader = labeling_loader

    # Train model
    train(run_fp, model, optimizer, train_loader, loss_fn, device, epochs, verbose=True)


if __name__ == "__main__":
    # Hyperparameters
    lr = 5e-4
    weight_decay = 1e-5
    batch_size = 64
    epochs = 500

    main(lr, weight_decay, batch_size, epochs)



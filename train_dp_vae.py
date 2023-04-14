import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from time import time
from itertools import product

from models import Encoder_VAE, Decoder_VAE, VAE
from data import load_latent
from utils import generate_run_id, get_input_args, Args

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

import opacus
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_vae(args, vae, train_loader, run_fp="runs_vae/test", kld_weight=0.0005, verbose=True):
    """Trains a VAE under DP using the Opacus library
    Use c_p as the clipping norm for the optimizer
    """
    os.makedirs(run_fp, exist_ok=True)

    # Check if anything inside run_fp exists
    if os.path.exists(f"{run_fp}/loss.txt"):
        print(f"{run_fp} already exists. Skipping...")
        return
    
    # Setup optimizer
    optimizer = optim.Adam(
        list(vae.parameters()),
        lr=args.lr,
        weight_decay=1e-5
    )

    # Make Private
    privacy_engine = PrivacyEngine()
    vae, optimizer, train_loader = privacy_engine.make_private(
        module=vae,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=args.c_p,
        noise_multiplier=args.noise_multiplier,
    )

    # Setup loss
    criterion = nn.MSELoss()

    # Track time
    start_time = time()
    eps = 0

    vae.train()
    with open(f"{run_fp}/loss.txt", "a") as f:
        for i in tqdm(range(args.n_g)):
            # Sample a batch
            data = next(iter(train_loader))[0].to(device)
        
            # Train with real data
            optimizer.zero_grad()
            output, mu, logvar, std = vae(data)

            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), 
                dim = 0
            )

            # Reshape output and data
            output = output.view(-1, 100)
            data = data.view(-1, 100)

            loss = criterion(output, data) + kld_loss
            loss.backward()
            optimizer.step()
            
            # print(f"{i}, {loss.item()}", file=f)
            
            if (i+1) % 200 == 0:
                # torch.save(vae._module.state_dict(), f"{run_fp}/vae_{i+1}.pt")
                # torch.save(privacy_engine.accountant, f"{run_fp}/accountant_{i+1}.pt")
                
                f.flush()
                if i < 10000 and eps < 300:
                    # Print epsilon every 200 iterations
                    eps = privacy_engine.get_epsilon(1e-6)
                    print(f"Epsilon {i+1} {eps}", file=f)
                    print(f"Epsilon {i+1} {eps}")

    print(f"Training took {time() - start_time} seconds")


def main(args, private=True, latent_type="ae_enc"):
    # Random Seeding
    torch.manual_seed(0)
    np.random.seed(0)

    if args.activation == "LeakyReLU":
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif args.activation == "Tanh":
        activation = nn.Tanh()

    # Models
    enc = Encoder_VAE(
        hidden_sizes=args.hidden, 
        latent_size=args.nz, 
        activation=activation
    ).to(device)
    dec = Decoder_VAE(
        hidden_sizes=args.hidden,
        latent_size=args.nz,
        activation=activation
    ).to(device)
    vae = VAE(enc, dec).to(device)

    # Privacy Validation
    ModuleValidator.validate(vae, strict=True)

    # Generate Run ID
    run_id = generate_run_id(args)

    # Load data
    if latent_type == "ae_enc":
        run_id = "ae-enc_" + run_id
        train_loader = load_latent(args.batch_size, 
            data_fp="data/ae_enc_latent_dataset.pt")
    elif latent_type == "ae_grad":
        run_id = "ae-grad_" + run_id
        train_loader = load_latent(args.batch_size, 
            data_fp="data/ae_grad_latent_dataset.pt")
    elif latent_type == "wgan":
        run_id = "wgan_" + run_id
        train_loader = load_latent(args.batch_size, 
            data_fp="data/wgan_latent_dataset.pt")
    else:
        raise ValueError("Latent type not supported")

    run_fp = os.path.join('runs_vae_eps/', run_id)
    os.makedirs(run_fp, exist_ok=True)

    print(f"================= Run ID: {run_id} =================")
    
    verbose = False
    train_vae(args, vae, train_loader, run_fp=run_fp, verbose=verbose)


def grid_search():
    # Private model Hyperparameter Search
    hiddens = [64]
    noise_multipliers = [0.2, 0.3, 0.4, 0.5] # [0.01, 0.05, 0.1]
    activations = ["LeakyReLU",]
    c_ps = [0.1] # [0.05, 0.01, 0.005, 0.001]
    lrs = [0.01] #  0.005]

    nz = 32
    n_d = 0
    n_g = 10000
    batch_size = 64
    
    for activation, c_p, noise_multiplier, lr in product(
            activations, c_ps, noise_multipliers, lrs):
        args = Args(
            # Model Parameters
            hidden=[64], nz=32, ngf=32, nc=1, activation=activation,
            # Privacy Parameters
            epsilon=50.0, delta=1e-6, noise_multiplier=noise_multiplier, c_p=c_p,
            # Training Parameters
            lr=lr, beta1=0.5, batch_size=batch_size, n_d=0, n_g=n_g, lambda_gp=0.0
        )
        # main(args, latent_type="ae_enc")
        main(args, latent_type="ae_grad")
    
    exit(0)



    # Private model Hyperparameter Search
    hiddens = [64]
    noise_multipliers = [0.2, 0.3, 0.4, 0.5] # [0.01, 0.05, 0.1]
    activations = ["LeakyReLU", "Tanh", ]
    c_ps = [0.5, 0.25, 0.1] # [0.05, 0.01, 0.005, 0.001]
    lrs = [0.01] #  0.005]

    nz = 32
    n_d = 0
    n_g = 48000
    batch_size = 64
    
    for activation, c_p, noise_multiplier, lr in product(
            activations, c_ps, noise_multipliers, lrs):
        args = Args(
            # Model Parameters
            hidden=[64], nz=32, ngf=32, nc=1, activation=activation,
            # Privacy Parameters
            epsilon=50.0, delta=1e-6, noise_multiplier=noise_multiplier, c_p=c_p,
            # Training Parameters
            lr=lr, beta1=0.5, batch_size=batch_size, n_d=0, n_g=n_g, lambda_gp=0.0
        )
        # main(args, latent_type="ae_enc")
        main(args, latent_type="ae_grad")


    # Non-private
    # # for lr in [0.01, 0.05, 0.1]:
    # #     for n_g in [1000, 10000, 50000]:
    
    # for c_p in c_ps:
    #     args = Args(
    #         # Model Parameters
    #         hidden=[64], nz=32, ngf=32, nc=1, activation="LeakyReLU",
    #         # Privacy Parameters
    #         epsilon=50.0, delta=1e-6, noise_multiplier=0.0, c_p=c_p,
    #         # Training Parameters
    #         lr=lr, beta1=0.5, batch_size=64, n_d=0, n_g=50000, lambda_gp=0.0
    #     )
    #     # main(args, latent_type="wgan")
    #     # main(args, latent_type="ae_enc")
    #     main(args, latent_type="ae_grad")


if __name__ == "__main__":
    grid_search()
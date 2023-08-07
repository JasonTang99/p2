import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
import os
from time import time

from utils import generate_run_id, get_input_args, Args
from models import Discriminator_FC, Generator_FC, Weight_Clipper, G_weights_init
from data import load_latent
from privacy import compute_ReLU_bounds, compute_Tanh_bounds, compute_empirical_bounds
from train_dpgan import train_WGAN

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


def main(args, private=True, c_g_mult=1.0, latent_type="ae_enc"):
    
    # Random Seeding
    torch.manual_seed(0)
    np.random.seed(0)

    # Setup models
    if args.activation == "LeakyReLU":
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif args.activation == "Tanh":
        activation = nn.Tanh()
    else:
        raise ValueError("Activation function not supported")
    
    netD = Discriminator_FC(
        hidden_sizes=args.hidden, 
        input_size=100, 
        activation=activation
    ).to(device)    
    netG = Generator_FC(
        nz=args.nz, 
        hidden_sizes=args.hidden, 
        output_size=(100,),
        sigmoid=False
    ).to(device)

    print(netD)
    print(netG)

    # Privacy Validation
    ModuleValidator.validate(netD, strict=True)

    c_g = 0
    if args.activation == "LeakyReLU":
        c_g = compute_ReLU_bounds(netD, args.c_p, input_size=(100,))
    elif args.activation == "Tanh":
        c_g = compute_Tanh_bounds(netD, args.c_p, input_size=(100,))

    # Use empirical c_g
    emp_c_g = compute_empirical_bounds(netD, args.c_p, input_size=(100,))
    # c_g = c_g_mult * emp_c_g
    print("ReLU Tanh:", c_g)
    print("Empirical c_g:", emp_c_g)
    c_g = max(c_g, emp_c_g)

    print("c_g:", c_g)
    
    # Setup optimizers
    weight_decay = 1e-5
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.9), weight_decay=weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.9), weight_decay=weight_decay)

    # Setup MNIST dataset using load_latent

    # Generate Run ID
    run_id = generate_run_id(args)
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
    
    run_fp = os.path.join('runs_latent/', run_id)
    os.makedirs(run_fp, exist_ok=True)

    print(f"================= Run ID: {run_id} =================")
    
    verbose = False
    # verbose = True
    train_WGAN(run_fp, args, netD, netG, optimizerD, optimizerG, train_loader, device, 
        c_g, private, verbose=verbose)


def grid_search():
    # Private model Hyperparameter Search
    hiddens = [128, 96, 32]
    noise_multipliers = [0.0, 0.05]
    activations = ["Tanh", "LeakyReLU", ]
    n_ds = [1, 2, 3]
    c_ps = [0.001, 0.005, 0.01]
    lrs = [5e-5, 1e-5, 5e-6]

    nz = 64
    latent_types = ["ae_enc", "ae_grad", "wgan"]

    # Randomly choose hyperparameters
    for i in range(1000000):
        np.random.seed(i)
        # Select random hyperparameter index
        hidden = list(np.random.choice(hiddens, 1))
        noise_multiplier = np.random.choice(noise_multipliers)
        activation = np.random.choice(activations)
        n_d = np.random.choice(n_ds)
        c_p = np.random.choice(c_ps)
        lr = np.random.choice(lrs)

        latent_type = np.random.choice(latent_types)

        args = Args(
            # Model Parameters
            hidden=hidden, nz=nz, ngf=32, nc=1, activation=activation,
            # Privacy Parameters
            epsilon=50.0, delta=1e-6, noise_multiplier=noise_multiplier, c_p=c_p,
            # Training Parameters
            lr=lr, beta1=0.5, batch_size=64, n_d=n_d, n_g=48000, lambda_gp=0.0
        )
        main(args, c_g_mult=1.0, latent_type=latent_type, private=noise_multiplier > 0)


if __name__ == "__main__":
    # Collect all parameters
    # args = get_input_args()
    # main(args)

    grid_search()


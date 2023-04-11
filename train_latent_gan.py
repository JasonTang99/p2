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

def main(args, private=True, c_g_mult=1.0):
    # Random Seeding
    torch.manual_seed(0)
    np.random.seed(0)

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)

    # Generate Run ID
    run_id = generate_run_id(args)
    if not private:
        if use_public_data:
            run_id = "public_" + run_id
        else:
            run_id = "private_" + run_id

    run_fp = os.path.join('runs_latent/', run_id)
    os.makedirs(run_fp, exist_ok=True)

    # Setup models
    if args.activation == "LeakyReLU":
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif args.activation == "Tanh":
        activation = nn.Tanh()
    else:
        raise ValueError("Activation function not supported")
    
    netD = Discriminator_FC(args.hidden, input_size=100, activation=activation).to(device)
    print(netD)
    
    netG = Generator_FC(nz=32, hidden_sizes=[16, 32], output_size=100).to(device)
    netG.apply(G_weights_init)

    # Privacy Validation
    ModuleValidator.validate(netD, strict=True)

    c_g = 0
    if args.activation == "LeakyReLU":
        c_g = compute_ReLU_bounds(netD, args.c_p)
    elif args.activation == "Tanh":
        c_g = compute_Tanh_bounds(netD, args.c_p)

    # Use empirical c_g
    if c_g == 0:
        emp_c_g = compute_empirical_bounds(netD, args.c_p)
        c_g = c_g_mult * emp_c_g
    print("Gradient clip:", c_g)
    
    # Setup optimizers
    weight_decay = 1e-5
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=weight_decay)

    # Setup MNIST dataset using load_MNIST
    labeling_loader, public_loader, private_loader, test_loader = load_MNIST(args.batch_size)
    train_loader = private_loader

    if private:
        # Setup Privacy Engine
        privacy_engine = PrivacyEngine()
        netD, optimizerD, train_loader = privacy_engine.make_private(
            module=netD,
            optimizer=optimizerD,
            data_loader=train_loader,
            max_grad_norm=c_g,
            noise_multiplier=args.noise_multiplier,
        )
        print(
            f"Model:{type(netD)}, \nOptimizer:{type(optimizerD)}, \nDataLoader:{type(train_loader)}"
        )
    else:
        privacy_engine = None
    
    verbose = False
    # verbose = True
    train_WGAN(run_fp, args, netD, netG, optimizerD, optimizerG, train_loader, device, 
        privacy_engine, verbose=verbose)


def train_non_private():
    # Non-private model on private data
    args = Args(
        # Model Parameters
        hidden=None, nz=100, ngf=32, nc=1, activation="LeakyReLU",
        # Privacy Parameters
        epsilon=float("inf"), delta=1e-6, noise_multiplier=0.0, c_p=0.01, 
        # Training Parameters
        lr=1e-4, beta1=0.5, batch_size=64, n_d=5, n_g=int(1e5), lambda_gp=0.0
    )
    main(args, private=False)

def grid_search():
    # Private model Hyperparameter Search
    hiddens = [[16, 12], ]# [12, 4, 4]]
    noise_multipliers = [0.05, 0.1, 0.2]
    activations = ["LeakyReLU", "Tanh",]
    lambda_gps = [0.0, ]

    for hidden in hiddens:
        for noise_multiplier in noise_multipliers:
            for activation in activations:
                for lambda_gp in lambda_gps:
                    args = Args(
                        # Model Parameters
                        hidden=hidden, nz=100, ngf=32, nc=1, activation=activation,
                        # Privacy Parameters
                        epsilon=50.0, delta=1e-6, noise_multiplier=noise_multiplier, c_p=0.01, 
                        # Training Parameters
                        lr=1e-4, beta1=0.5, batch_size=64, n_d=5, n_g=int(2e5), lambda_gp=lambda_gp
                    )
                    main(args, c_g_mult=2.0)

if __name__ == "__main__":
    # Collect all parameters
    # args = get_input_args()
    # main(args)

    train_non_private()
    # grid_search()


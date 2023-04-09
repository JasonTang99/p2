import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
import os
from time import time

from utils import generate_run_id, get_input_args, Args
from models import Discriminator, Generator_MNIST, Weight_Clipper, G_weights_init
from data import load_MNIST
from privacy import compute_ReLU_bounds, compute_Tanh_bounds

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


def train(run_fp, args, netD, netG, optimizerD, optimizerG, train_loader, device, privacy_engine=None, verbose=False):
    """Training process
    if privacy_engine is not None, then train with DP
    """

    # Check if anything inside run_fp exists
    if os.path.exists(f"{run_fp}/loss.txt"):
        print(f"{run_fp} already exists. Skipping...")
        return
    
    # Track time
    start_time = time()
    print_mod = 10
    
    netD.train()
    netG.train()
    clipper = Weight_Clipper(args.c_p)
    
    with open(f"{run_fp}/loss.txt", "a") as f:
        for i in tqdm(range(args.n_g)):
            # Update Discriminator
            if privacy_engine is not None:
                netD.enable_hooks()
            netD.train()
            netG.eval()
            for j in range(args.n_d):
                # Train with real data
                real_data = next(iter(train_loader))[0].to(device)
                real_data = real_data.view(real_data.size(0), -1)
                
                # Train with fake data
                noise = torch.randn(real_data.size(0), 100, 1, 1).to(device)
                fake_data = netG(noise)

                # Run Discriminator
                real_output = netD(real_data)
                fake_output = netD(fake_data)

                # Calculate loss (Wasserstein Loss)
                d_loss = -torch.mean(real_output) + torch.mean(fake_output)

                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()

                # Clip weights in discriminator
                netD.apply(clipper)

                if verbose:
                    print(f"Epoch: {i} ({j}/{args.n_d}) D_loss: {d_loss.item()} \
                        eps: {privacy_engine.get_epsilon(args.delta) if privacy_engine is not None else 0}")
                if i % print_mod == 0:
                    print(f"{i}.{j}, {d_loss.item()}", file=f)

            if privacy_engine is not None:
                netD.disable_hooks()
            netD.eval()
            netG.train()

            # Update Generator
            noise = torch.randn(args.batch_size, 100, 1, 1).to(device)
            fake_output = netD(netG(noise))
            g_loss = -torch.mean(fake_output)

            # Update Generator
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            if verbose:
                print(f"Epoch: {i} G_loss: {g_loss.item()}")
                # print eps
                if privacy_engine is not None:
                    print(f"Epoch: {i} eps: {privacy_engine.get_epsilon(args.delta)}")
            if i % print_mod == 0:
                print(f"{i}, {g_loss.item()}", file=f)

            if (i+1) % 2000 == 0:
                # Non-private model
                if privacy_engine is None:
                    torch.save(netG.state_dict(), f"{run_fp}/netG_{i+1}.pt")
                    torch.save(netD.state_dict(), f"{run_fp}/netD_{i+1}.pt")
                    continue
                
                # Private model
                # eps = privacy_engine.get_epsilon(args.delta)
                # print(f"Saving model at iteration {i+1}, epsilon {eps}")
                # print(f"{i+1}: epsilon {eps}", file=f)
                
                print(f"{i+1} Training time: {time() - start_time}", file=f)
                torch.save(netG.state_dict(), f"{run_fp}/netG_{i+1}.pt")
                torch.save(netD._module.state_dict(), f"{run_fp}/netD_{i+1}.pt")
                torch.save(privacy_engine.accountant, f"{run_fp}/accountant_{i+1}.pth")
        
        # Save train time
        print(f"Training time: {time() - start_time}", file=f)
        print(f"Training time: {time() - start_time}")


def main(args, private=True, use_public_data=False):
    # Random Seeding
    torch.manual_seed(0)
    np.random.seed(0)

    # Print arguments
    print(args)

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate Run ID
    run_id = generate_run_id(args)
    if not private:
        if use_public_data:
            run_id = "public_" + run_id
        else:
            run_id = "private_" + run_id

    run_fp = os.path.join('runs/', run_id)
    os.makedirs(run_fp, exist_ok=True)

    # Setup models
    netD = Discriminator(args.hidden, input_size=784).to(device)
    netG = Generator_MNIST(nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
    netG.apply(G_weights_init)

    # Privacy Validation
    ModuleValidator.validate(netD, strict=True)

    # Setup parameters for Gradient Clip Calculation
    c_g = compute_ReLU_bounds(netD, args.c_p)
    print("Gradient clip:", c_g)

    # Setup optimizers
    weight_decay = 1e-5
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=weight_decay)

    # Setup MNIST dataset using load_MNIST
    labeling_loader, public_loader, private_loader, test_loader = load_MNIST(args.batch_size)
    
    if use_public_data:
        train_loader = public_loader
    else:
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

        verbose = False
        # verbose = True
        train(run_fp, args, netD, netG, optimizerD, optimizerG, train_loader, device, privacy_engine, verbose=verbose)
    else:
        verbose = False
        # verbose = True
        train(run_fp, args, netD, netG, optimizerD, optimizerG, train_loader, device, privacy_engine=None, verbose=verbose)


if __name__ == "__main__":
    # Collect all parameters
    args = get_input_args()

    # Non-private model
    args = Args(
        # Model Parameters
        hidden=[16, 12], nz=100, ngf=32, nc=1, 
        # Privacy Parameters
        epsilon=float("inf"), delta=1e-6, noise_multiplier=0.4, c_p=0.005, 
        # Training Parameters
        lr=1e-4, beta1=0.5, batch_size=64, n_d=4, n_g=int(5e5)
    )
    main(args, private=False, public=True)
    main(args, private=False, public=False)

    # Private model Hyperparameter Search
    hiddens = [[16, 12], [12, 4, 4]]
    noise_multipliers = [0.6, 0.4]

    for hidden in hiddens:
        for noise_multiplier in noise_multipliers:
            args = Args(
                # Model Parameters
                hidden=hidden, nz=100, ngf=32, nc=1, 
                # Privacy Parameters
                epsilon=50.0, delta=1e-6, noise_multiplier=noise_multiplier, c_p=0.005, 
                # Training Parameters
                lr=1e-4, beta1=0.5, batch_size=32, n_d=4, n_g=int(5e5)
            )
            main(args)

    # args = Args(
    #     # Model Parameters
    #     hidden=[16, 12], nz=100, ngf=32, nc=1, 
    #     # Privacy Parameters
    #     epsilon=50.0, delta=1e-6, noise_multiplier=0.4, c_p=0.005, 
    #     # Training Parameters
    #     lr=5e-4, beta1=0.5, batch_size=64, n_d=4, n_g=int(1e4)
    # )

    # main(args)




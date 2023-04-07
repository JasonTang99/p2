import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
import os
from time import time

from utils import generate_run_id, get_input_args, Args
from models import Discriminator, Generator_MNIST, Weight_Clipper, G_weights_init
from data import MNIST

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


# Given parameter clip bounds c_p, compute maximal ReLU activation bounds B_sigma
def compute_ReLU_bounds(model, c_p, input_size=(784,), input_bounds=1.0, B_sigma_p=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.ones(input_size[0]).to(device) * input_bounds
    B_sigma = 0.0
    sum_mk_mkp1 = 0
    skip_first = True

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            W = torch.ones_like(layer.weight) * c_p
            sample = W @ sample
            
            B_sigma = max(B_sigma, sample.max().detach().item())
            
            if skip_first:
                skip_first = False
            else:
                sum_mk_mkp1 += W.shape[0] * W.shape[1]
    
    c_g = 2 * c_p * B_sigma * (B_sigma_p ** 2) * sum_mk_mkp1
    print("B_sigma", B_sigma)
    print("sum_mk_mkp1", sum_mk_mkp1)
    print("c_g", c_g)
    return c_g


def train(run_fp, args, netD, netG, optimizerD, optimizerG, train_loader, device, privacy_engine=None, verbose=False):
    """Training process
    if privacy_engine is not None, then train with DP
    """

    # Check if anything inside run_fp exists
    # if os.path.exists(f"{run_fp}/loss.txt"):
    #     print(f"{run_fp} already exists. Skipping...")
    #     return
    
    # Track time
    start_time = time()
    print_mod = 10
    
    netD.train()
    netG.train()
    clipper = Weight_Clipper(args.c_p)
    
    with open(f"{run_fp}/loss.txt", "a") as f:
        for i in tqdm(range(args.n_g)):
            # Update Discriminator
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
                # for p in netD.parameters():
                #     p.data.clamp_(-args.c_p, args.c_p)
                netD.apply(clipper)

                if verbose:
                    print(f"Epoch: {i} ({j}/{args.n_d}) D_loss: {d_loss.item()} \
                        eps: {privacy_engine.get_epsilon(args.delta) if privacy_engine is not None else 0}")
                if i % print_mod == 0:
                    print(f"{i}.{j}, {d_loss.item()}", file=f)

            netD.eval()
            netD.disable_hooks()
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
            if i % print_mod == 0:
                print(f"{i}, {g_loss.item()}", file=f)

            if (i+1) % 2000 == 0:
                # Non-private model
                if privacy_engine is None:
                    torch.save(netG.state_dict(), f"{run_fp}/netG_{i+1}.pt")
                    torch.save(netD.state_dict(), f"{run_fp}/netD_{i+1}.pt")
                    continue
                
                # Private model
                eps = privacy_engine.get_epsilon(args.delta)
                print(f"Saving model at iteration {i+1}, epsilon {eps}")
                print(f"{i+1}: epsilon {eps}", file=f)
                torch.save(netG.state_dict(), f"{run_fp}/netG_{i+1}.pt")
                torch.save(netD._module.state_dict(), f"{run_fp}/netD_{i+1}.pt")
                torch.save(privacy_engine.accountant, f"{run_fp}/accountant_{i+1}.pth")
        
        # Save train time
        print(f"Training time: {time() - start_time}", file=f)
        print(f"Training time: {time() - start_time}")


def main(args):
    # Random Seeding
    torch.manual_seed(0)
    np.random.seed(0)

    # Print arguments
    print(args)

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate Run ID
    run_id = generate_run_id(args)
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
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Setup MNIST dataset
    train_loader, _ = MNIST(args.batch_size)


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
    verbose = True
    train(run_fp, args, netD, netG, optimizerD, optimizerG, train_loader, device, privacy_engine, verbose=verbose)

if __name__ == "__main__":
    # Collect all parameters
    # args = get_input_args()
    # or
    args = Args(
        # Model Parameters
        hidden=[16, 12], nz=100, ngf=32, nc=1, 
        # Privacy Parameters
        epsilon=50.0, delta=1e-6, noise_multiplier=0.4, c_p=0.005, 
        # Training Parameters
        lr=5e-4, beta1=0.5, batch_size=64, n_d=3, n_g=int(1e4)
    )

    main(args)





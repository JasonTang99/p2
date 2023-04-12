import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
import os
from time import time

from utils import generate_run_id, get_input_args, Args
from models import Discriminator_FC, Discriminator_MNIST, Generator_MNIST, Weight_Clipper, G_weights_init, Generator_FC
from data import load_MNIST
from privacy import compute_ReLU_bounds, compute_Tanh_bounds, compute_empirical_bounds

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.autograd as autograd

import torchvision
import torchvision.transforms as transforms

import opacus
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import warnings
warnings.filterwarnings("ignore")


def train_WGAN(run_fp, args, netD, netG, optimizerD, optimizerG, train_loader, device, 
        privacy_engine=None, verbose=False):
    """Training process
    if privacy_engine is not None, then train with DP
    improved: if True, use improved WGAN training process
    """
    # Check if anything inside run_fp exists
    if os.path.exists(f"{run_fp}/loss.txt"):
        print(f"{run_fp} already exists. Skipping...")
        return
    
    # Track time
    start_time = time()
    print_mod = 10
    save_mod = 2000
    clipper = Weight_Clipper(args.c_p)
    
    with open(f"{run_fp}/loss.txt", "a") as f:
        for i in tqdm(range(args.n_g)):
            # Update Discriminator_FC
            if privacy_engine is not None:
                netD.enable_hooks()
            netD.train()
            netG.eval()

            for j in range(args.n_d):
                # Generate real and fake
                real_data = next(iter(train_loader))[0].to(device)
                noise = torch.randn(real_data.size(0), args.nz).to(device)
                fake_data = netG(noise)

                # Run Discriminator_FC
                real_output = netD(real_data)
                fake_output = netD(fake_data)

                # Calculate loss
                if args.lambda_gp == 0.0:
                    # Standard WGAN loss
                    d_loss = -torch.mean(real_output) + torch.mean(fake_output)
                else:
                    print("Using improved WGAN loss")
                    # Improved WGAN-GP loss
                    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
                    alpha = alpha.expand(real_data.size())

                    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
                    interpolates = autograd.Variable(interpolates, requires_grad=True)
                    disc_interpolates = netD(interpolates)

                    gradients = autograd.grad(
                        outputs=disc_interpolates,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(disc_interpolates),
                        create_graph=True, 
                        retain_graph=True, 
                        only_inputs=True
                    )[0]

                    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lambda_gp
                    
                    # eps = torch.rand(real_data.size(0), 1, 1, 1).to(device)
                    # x_hat = eps * real_data + (1 - eps) * fake_data
                    # x_hat_output = netD(x_hat)

                    # grad_x_hat = torch.autograd.grad(
                    #     outputs=x_hat_output, 
                    #     inputs=x_hat,
                    #     grad_outputs=torch.ones_like(x_hat_output), 
                    #     create_graph=False
                    # )[0]
                    # grad_x_hat_norm = torch.sqrt(torch.sum(
                    #     grad_x_hat.view(grad_x_hat.size(0), -1) ** 2, 
                    #     dim=1
                    # ))
                    # grad_penalty = args.lambda_gp * torch.mean((grad_x_hat_norm - 1) ** 2)
                    
                    d_loss = -torch.mean(real_output) + torch.mean(fake_output) + grad_penalty

                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()

                # Clip weights in discriminator (if not using WGAN-GP)
                if args.lambda_gp == 0.0:
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
            noise = torch.randn(args.batch_size, args.nz).to(device)
            fake_output = netD(netG(noise))
            g_loss = -torch.mean(fake_output)

            # Update Generator
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            if verbose:
                print(f"Epoch: {i} G_loss: {g_loss.item()} \
                    eps: {privacy_engine.get_epsilon(args.delta) if privacy_engine is not None else 0}")
            if i % print_mod == 0:
                print(f"{i}, {g_loss.item()}", file=f)

            if (i+1) % save_mod == 0:
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


def main(args, private=True, use_public_data=False, c_g_mult=1.0):
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

    run_fp = os.path.join('runs_gen_fc_3/', run_id)
    os.makedirs(run_fp, exist_ok=True)

    # Setup models
    if args.activation == "LeakyReLU":
        activation = nn.LeakyReLU(0.2, inplace=True)
    elif args.activation == "Tanh":
        activation = nn.Tanh()
    else:
        # Throw error
        raise ValueError("Activation function not supported")
    
    if args.hidden is not None:
        netD = Discriminator_FC(args.hidden, input_size=784, activation=activation).to(device)
    else:
        # Use CNN
        netD = Discriminator_MNIST(ndf=16, nc=args.nc, activation=activation).to(device)
    print(netD)

    # netG = Generator_MNIST(nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
    # netG.apply(G_weights_init)
    netG = Generator_FC(hidden_sizes=[256], nz=args.nz, output_size=(1, 28, 28)).to(device)

    # For Latent Space 
    # D = Discriminator_FC(hidden_sizes=[16, 16], input_size=100).to(device)
    # G = Generator_FC(nz=32, hidden_sizes=[16, 32], output_size=100).to(device)

    # Privacy Validation
    ModuleValidator.validate(netD, strict=True)

    if args.activation == "LeakyReLU":
        c_g = compute_ReLU_bounds(netD, args.c_p)
    elif args.activation == "Tanh":
        c_g = compute_Tanh_bounds(netD, args.c_p)

    # Use empirical c_g
    emp_c_g = compute_empirical_bounds(netD, args.c_p)
    c_g = c_g_mult * emp_c_g

    print("Gradient clip:", c_g)
    
    # Setup optimizers
    weight_decay = 1e-6
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.9), weight_decay=weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.9), weight_decay=weight_decay)

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
    else:
        privacy_engine = None
    
    verbose = False
    # verbose = True
    train_WGAN(run_fp, args, netD, netG, optimizerD, optimizerG, train_loader, device, 
        privacy_engine, verbose=verbose)


def train_non_private():
    args = Args(hidden=[256], nz=100, ngf=32, nc=1, epsilon='inf', delta=1e-06, noise_multiplier=0.0, c_p=0.01, lr=5e-05, beta1=0.0, batch_size=64, n_d=3, n_g=int(5e5), activation='LeakyReLU', lambda_gp=0.0)
    main(args, private=False, use_public_data=True)

    exit(0)

    # Non-private model on private data
    # lambda_gps = [1.0, 0.0]
    # hiddens = [[128], None]    
    # c_ps = [0.02, 0.01, 0.005]
    # n_ds = [5, 3]
    # n_zs = [100, 50]
    # lrs = [5e-5, 5e-4]

    lambda_gps = [0.0]
    hiddens = [[128, 128]]    
    c_ps = [0.01, 0.005]
    n_ds = [4]
    n_zs = [100]
    lrs = [1e-4]

    # for hidden in hiddens:
    #     for lambda_gp in lambda_gps:
    #         args = Args(
    #             # Model Parameters
    #             hidden=hidden, nz=100, ngf=32, nc=1, activation="LeakyReLU",
    #             # Privacy Parameters
    #             epsilon=float("inf"), delta=1e-6, noise_multiplier=0.0, c_p=0.01, 
    #             # Training Parameters
    #             lr=1e-4, beta1=0.5, batch_size=64, n_d=5, n_g=int(1e5), lambda_gp=lambda_gp
    #         )
    #         main(args, private=False, use_public_data=False)


    # Non-private model on public data (using improved WGAN)
    for hidden in hiddens:
        for n_z in n_zs:
            for lr in lrs:
                for n_d in n_ds:
                    for lambda_gp in lambda_gps:
                        for c_p in c_ps:
                            if lambda_gp != 0.0:
                                c_p = 0.0
                            
                            args = Args(
                                # Model Parameters
                                hidden=hidden, nz=n_z, ngf=32, nc=1, activation="LeakyReLU",
                                # Privacy Parameters
                                epsilon=float("inf"), delta=1e-6, noise_multiplier=0.0, c_p=c_p, 
                                # Training Parameters
                                lr=lr, beta1=0.0, batch_size=64, n_d=n_d, n_g=int(2e5), lambda_gp=lambda_gp
                            )
                            main(args, private=False, use_public_data=True)

def grid_search():
    # Private model Hyperparameter Search
    hiddens = [None, ]# [16, 12], ]# [12, 4, 4]]
    noise_multipliers = [0.0, 0.05, 0.1, 0.2]
    activations = ["LeakyReLU",] # "Tanh",]
    lambda_gps = [0.0, 10.0]

    for hidden in hiddens:
        for noise_multiplier in noise_multipliers:
            for activation in activations:
                for lambda_gp in lambda_gps:
                    args = Args(
                        # Model Parameters
                        hidden=hidden, nz=100, ngf=32, nc=1, activation=activation,
                        # Privacy Parameters
                        epsilon=38.0, delta=1e-6, noise_multiplier=noise_multiplier, c_p=0.01, 
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


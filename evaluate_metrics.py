import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
import os
from time import time, sleep
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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

from utils import generate_run_id, get_input_args, Args, parse_run_id
from models import Discriminator_FC, Generator_MNIST, Weight_Clipper, G_weights_init, Generator_FC, Decoder_Mini, Encoder_VAE, Decoder_VAE, VAE
from data import load_MNIST
from metrics import get_IS, get_FID
from model_inversion import enc_fp, dec_fp, gen_fp

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def last_num_models(run_fp, num=10, query="netG"):
    # Filter out non-model files "netG_*.pt"
    models = [model for model in os.listdir(run_fp) if model.startswith(query)]
    models = sorted(models, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    models = models[-num:]
    return models

# Calculate FIDs of last 10 models for latent GAN using WGAN
def calculate_FIDs_WGAN(args, run_fp):
    # Load Public WGAN Generator
    pub_G = Generator_FC(hidden_sizes=[256], nz=100).to(device)
    pub_G.load_state_dict(torch.load(gen_fp))

    # Consider final models
    models = last_num_models(run_fp, num=num)

    # Calculate FIDs
    FIDs = []
    for model in models:
        model_fp = os.path.join(run_fp, model)
        print("Loading {}".format(model_fp))

        G = Generator_FC(
            hidden_sizes=args.hidden,
            nz=args.nz,
            output_size=(100,)
        ).to(device)
        G.load_state_dict(torch.load(model_fp))
        G.eval()

        # Generate 2048 fake images
        noise = torch.randn(2048, args.nz).to(device)
        fake = pub_G(G(noise))
        fake = fake.view(fake.size(0), 1, 28, 28)

        # Calculate Frechet Inception Distance
        # FID = get_FID(fake)
        FID = get_IS(fake)
        FIDs.append((model_fp, FID))
        print("Frechet Inception Distance:", FID)

    # Return Best model_fp
    # best_model_fp, best_FID = min(FIDs, key=lambda x: x[1])
    best_model_fp, best_FID = max(FIDs, key=lambda x: x[1])

    return best_model_fp, best_FID

def calculate_AE(args, run_fp, metric="IS"):
    # Load Public Decoder
    pub_Dec = Decoder_Mini(latent_size=100).to(device)
    pub_Dec.load_state_dict(torch.load(dec_fp))

    # Consider final model
    # model = last_num_models(run_fp, num=1, query="vae")[0]
    # Consider halfway model 28000 steps
    model = "vae_24000.pt"
    gen_fp = os.path.join(run_fp, model)
    print("Loading {}".format(gen_fp))

    vae = VAE(
        Encoder_VAE(args.hidden, latent_size=args.nz), 
        Decoder_VAE(args.hidden, latent_size=args.nz)
    ).to(device)
    vae.load_state_dict(torch.load(gen_fp))
    G = vae.decoder
    G.eval()

    # Generate 2048 fake images
    noise = torch.randn(2048, args.nz).to(device)
    fake = pub_Dec(G(noise))
    fake = fake.view(fake.size(0), 1, 28, 28)

    # Calculate Metric
    if metric == "IS":
        metric = get_IS(fake)
    elif metric == "FID":
        metric = get_FID(fake)
    
    return metric


if __name__ == "__main__":
    from model_inversion import enc_fp, dec_fp, gen_fp
    pub_G = Generator_FC(hidden_sizes=[256], nz=100).to(device)
    pub_G.load_state_dict(torch.load(gen_fp))

    pub_Dec = Decoder_Mini(latent_size=100).to(device)
    pub_Dec.load_state_dict(torch.load(dec_fp))
    
    # Get valid run_ids
    folder = "runs_vae"

    from itertools import product
    # noise_multipliers = [0.3, 0.4, 0.5] 
    # activations = ["LeakyReLU", "Tanh", ]
    # c_ps = [0.5, 0.1, 0.01] 
    # lrs = [0.01] 
    hiddens = [64]
    noise_multipliers = [0.3, 0.4, 0.5, 0.6] 
    activations = ["LeakyReLU", "Tanh", ]
    c_ps = [0.5, 0.25, 0.1, 0.05] 
    lrs = [0.01, 0.02] 
    nz = 32
    n_d = 0
    n_g = 48000
    batch_size = 64
    

    valid_run_ids = []
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
        run_id = "ae-grad_" + generate_run_id(args)
        print(run_id, os.path.exists("runs_vae/{}".format(run_id)))
        if os.path.exists("runs_vae/{}".format(run_id)):
            valid_run_ids.append(run_id)
    print(len(valid_run_ids))
    # assert len(valid_run_ids) == 18

    # Save results to a pandas dataframe
    df = pd.DataFrame(columns=[
        "hidden", "noise_multiplier", "activation", 
        "n_g", "c_p", "lr", "IS", "latent_type", "model_fp"
    ])

    IS_fp = "results/vae_IS_final_half.csv"
    FID_fp = "results/vae_FID_final_half.csv"

    # Iterate over all runs and calculate IS
    if not os.path.exists(IS_fp):
        for run_id in tqdm(valid_run_ids):
            run_fp = os.path.join(folder, run_id)
            args = parse_run_id(run_id)
            print(args.noise_multiplier, args.c_p)

            # Calculate IS
            IS = calculate_AE(args, run_fp, metric="IS")

            # Save results
            df = df.append({
                "hidden": args.hidden,
                "noise_multiplier": args.noise_multiplier,
                "activation": args.activation,
                # "n_g": args.n_g,
                n_g: 24000,
                "c_p": args.c_p,
                "lr": args.lr,
                "IS": IS,
                "latent_type": "ae-grad",
                "model_fp": run_fp
            }, ignore_index=True)

        # Save results
        df.to_csv(IS_fp, index=False)

    sleep(5)

    # Save results to a pandas dataframe
    df = pd.DataFrame(columns=[
        "model_fp", "FID"
    ])
    # Iterate over all runs and calculate IS
    if not os.path.exists(FID_fp):
        for run_id in tqdm(valid_run_ids):
            run_fp = os.path.join(folder, run_id)
            args = parse_run_id(run_id)
            print(args.noise_multiplier, args.c_p)

            # Calculate FID
            FID = calculate_AE(args, run_fp, metric="FID")

            # Save results
            df = df.append({
                "model_fp": run_fp,
                "FID": FID
            }, ignore_index=True)

        # Save results
        df.to_csv(FID_fp, index=False)


    exit(0)

    # Compute for mode collapse
    wgan_latent_sample_fp = "wgan_32_64_32_1_50.0_1e-06_0.0_0.01_1e-05_0.5_64_1_48000_LeakyReLU_0.0"
    ae_enc_sample_fp = "ae-enc_96_64_32_1_50.0_1e-06_0.0_0.001_5e-05_0.5_64_1_48000_Tanh_0.0" 
    ae_grad_sample_fp = "ae-grad_96_64_32_1_50.0_1e-06_0.0_0.005_5e-05_0.5_64_1_48000_Tanh_0.0"

    metrics = []
    for run_id in [wgan_latent_sample_fp, ae_enc_sample_fp, ae_grad_sample_fp]:
        run_fp = os.path.join("runs_latent", run_id)
        print(run_id)

        run_id = run_id.split("/")[-1]
        args = parse_run_id(run_id)

        gen_fp = os.path.join(run_fp, 'netG_48000.pt')

        G = Generator_FC(args.hidden, args.nz, output_size=(100,)).to(device)
        G.load_state_dict(torch.load(gen_fp))
        G.eval()

        # Generate 2048 samples
        noise = torch.randn(2048, args.nz, device=device)
        print(f"noise: {noise.shape}")
        fake = G(noise)
        if run_id.startswith("wgan"):
            fake = pub_G(fake)
        else:
            fake = pub_Dec(fake)
        print(f"fake: {fake.shape}")

        # Calculate IS
        # IS = get_IS(fake)
        # print(f"IS: {IS}")

        # Calculate FID
        FID = get_FID(fake)
        print(f"FID: {FID}")

        # metrics.append((IS, FID))
    print(metrics)



    # for i in range(400000, 500000 + 1, 10000):
    #     gen_fp = os.path.join(run_fp, 'netG_{}.pt'.format(i))
    #     if os.path.exists(gen_fp):
    #         print("Loading {}".format(gen_fp))

    #         G = Generator_FC([128], args.nz).to(device)
    #         G.load_state_dict(torch.load(gen_fp))
    #         G.eval()
    
    #         # Generate 2048 fake images
    #         noise = torch.randn(2048, 100).to(device)
    #         fake = G(noise)
    #         fake = fake.view(fake.size(0), 1, 28, 28)

    #         # Calculate Inception Score
    #         IS = get_IS(fake)
    #         print("Inception Score:", IS)

            # Calculate Frechet Inception Distance
            # FID = get_FID(fake)
            # print("Frechet Inception Distance:", FID)

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

def calculate_AE(args, run_fp):
    # Load Public Decoder
    pub_Dec = Decoder_Mini(latent_size=100).to(device)
    pub_Dec.load_state_dict(torch.load(dec_fp))

    # Consider final model
    model = last_num_models(run_fp, num=1, query="vae")[0]
    gen_fp = os.path.join(run_fp, model)
    print("Loading {}".format(gen_fp))

    vae = VAE(
        Encoder_VAE(args.hidden, latent_size=args.nz), 
        Decoder_VAE(args.hidden, latent_size=args.nz)
    ).to(device)
    vae.load_state_dict(torch.load(gen_fp))
    vae.eval()

    G = vae.decoder
    G.eval()

    # Generate 2048 fake images
    noise = torch.randn(2048, args.nz).to(device)
    fake = pub_Dec(G(noise))
    fake = fake.view(fake.size(0), 1, 28, 28)

    # Calculate IS
    FID = get_FID(fake)
    # IS = get_IS(fake)

    return FID



if __name__ == "__main__":
    # Save results to a pandas dataframe
    # Cols:     hiddens = [[96], [64]]
    # noise_multipliers = [0.0, 0.1]
    # activations = ["Tanh", "LeakyReLU", ]
    # n_ds = [3, 5]
    # c_ps = [0.01, 0.02]

    df = pd.DataFrame(columns=[
        "hidden", "noise_multiplier", "activation", 
        "n_g", "c_p", "lr", "IS", "latent_type", "model_fp"
    ])

    folder = "runs_vae"
    valid_runs_ids = []
    for run_id in os.listdir("runs_vae"):
        run_fp = f"runs_vae/{run_id}"
        args = parse_run_id(run_id)
        if run_id.startswith("ae-grad") and \
                args.noise_multiplier > 0 and \
                args.n_g > 20000 and \
                args.n_g <= 50000:
            valid_runs_ids.append(run_id)
    valid_runs_ids = sorted(valid_runs_ids)

    # Iterate over all runs
    for run_id in tqdm(valid_runs_ids):
        run_fp = os.path.join(folder, run_id)
        args = parse_run_id(run_id)
        print(args.noise_multiplier, args.c_p)

        # Calculate FID
        FID = calculate_AE(args, run_fp)

        # Save results
        df = df.append({
            "hidden": args.hidden,
            "noise_multiplier": args.noise_multiplier,
            "activation": args.activation,
            "n_g": args.n_g,
            "c_p": args.c_p,
            "lr": args.lr,
            "FID": FID,
            "latent_type": "ae-grad",
            "model_fp": run_fp
        }, ignore_index=True)

    # Save results
    df.to_csv("vae_FID.csv", index=False)

    # Print results
    print(df)


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

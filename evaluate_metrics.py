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
from models import Discriminator_FC, Generator_MNIST, Weight_Clipper, G_weights_init, Generator_FC
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
def calculate_FIDs_WGAN(args, run_fp, num=10):
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

        # Sample 4 fake latents
        # noise = torch.randn(4, args.nz).to(device)
        # fake = G(noise)
        # print(fake)
        # continue

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


if __name__ == "__main__":
    # run_id = "/home/jason/p2/runs_gen_fc_3/public_128_100_32_1_inf_1e-06_0.0_0.01_5e-05_0.0_64_3_500000_LeakyReLU_0.0"
    # run_id = "/home/jason/p2/runs_gen_fc_3/public_256_100_32_1_inf_1e-06_0.0_0.01_5e-05_0.0_64_3_500000_LeakyReLU_0.0"
    
    
    # run_id = "runs_latent/ae-enc_64_64_32_1_50.0_1e-06_0.0_0.01_5e-05_0.5_64_3_100000_LeakyReLU_0.0"
    
    # Save results to a pandas dataframe
    # Cols:     hiddens = [[96], [64]]
    # noise_multipliers = [0.0, 0.1]
    # activations = ["Tanh", "LeakyReLU", ]
    # n_ds = [3, 5]
    # c_ps = [0.01, 0.02]
    df = pd.DataFrame(columns=[
        "hiddens", "noise_multiplier", "activation", 
        "n_d", "c_p", "FID", "model_fp"
    ])

    folder = "runs_latent"
    for run_id in os.listdir(folder):
        run_fp = os.path.join(folder, run_id)
        args = parse_run_id(run_id)

        # Calculate FIDs
        best_model_fp, best_FID = calculate_FIDs_WGAN(args, run_fp, num=1)

        # Save results
        df = df.append({
            "hiddens": args.hidden,
            "noise_multiplier": args.noise_multiplier,
            "activation": args.activation,
            "n_d": args.n_d,
            "c_p": args.c_p,
            "FID": best_FID,
            "model_fp": best_model_fp
        }, ignore_index=True)

    # Save results
    df.to_csv("latent_results.csv", index=False)

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

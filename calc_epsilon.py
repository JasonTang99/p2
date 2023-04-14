import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple
import os
from time import time, sleep
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
from models import Discriminator_FC, Generator_MNIST, Weight_Clipper, G_weights_init, Generator_FC, Encoder_Mini, Decoder_Mini, VAE, Encoder_VAE, Decoder_VAE
from data import load_MNIST
from metrics import get_IS, get_FID
from model_inversion import enc_fp, dec_fp, gen_fp
from evaluate_metrics import last_num_models

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from scipy.interpolate import interp1d
import pickle

def calculate_epsilon_used(run_fp, delta=1e-5):
    """Calculates the epsilon used for a given run_fp
    We need to linearly extrapolate past a few thousand batches since
        the accountant uses too much memory and gets killed
    """
    run_id = run_fp.split("/")[-1]
    args = parse_run_id(run_id)

    accts = sorted([
        (int(fp.split("_")[-1].strip(".pt")), fp) 
        for fp in os.listdir(run_fp) if fp.startswith("accountant")
    ])

    epsilons = []
    for batch_idx, acct_fp in accts[::2]:
        if batch_idx > 10000:
            break
        try:
            print(batch_idx, args.noise_multiplier, args.c_p, acct_fp)
            accountant = torch.load(f"{run_fp}/{acct_fp}")
            curr_eps = accountant.get_epsilon(delta)
            print(batch_idx, curr_eps)
        except:
            break
        # If epsilon is too high, we can't use it
        if curr_eps > 200:
            epsilons = []
            break            
        epsilons.append([batch_idx, curr_eps])
    # print(epsilons)

    # Linearly interpolate to get the epsilon used
    if len(epsilons) == 0:
        f = interp1d(np.array([0, 1]), np.array([-1, -1]), kind="linear", fill_value="extrapolate")
    else:
        epsilons = np.array(epsilons)
        f = interp1d(epsilons[:, 0], epsilons[:, 1], kind="linear", fill_value="extrapolate")
    
    # Pickle the function
    with open(f"{run_fp}/epsilon_used.pkl", "wb") as file:
        pickle.dump(f, file)
    
    return f(args.n_g)

# run_fp = "runs_vae/ae-grad_64_32_32_1_50.0_1e-06_0.2_0.5_0.01_0.5_64_0_100000_LeakyReLU_0.0"
# calculate_epsilon_used(run_fp)

valid_runs_fps = []
for run_id in os.listdir("runs_vae"):
    run_fp = f"runs_vae/{run_id}"
    args = parse_run_id(run_id)
    if run_id.startswith("ae-grad") and \
            args.noise_multiplier > 0 and \
            args.n_g > 20000 and \
            args.n_g <= 50000:
        valid_runs_fps.append(run_fp)

print(len(valid_runs_fps))

# Calculate the epsilon used for each run
for run_fp in tqdm(valid_runs_fps):
    # if not os.path.exists(f"{run_fp}/epsilon_used.pkl"):
    print(calculate_epsilon_used(run_fp))
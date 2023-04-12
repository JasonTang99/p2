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
from models import Discriminator_FC, Generator_MNIST, Weight_Clipper, G_weights_init, Generator_FC
from data import load_MNIST
from metrics import get_IS, get_FID

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")











if __name__ == "__main__":
    run_id = "/home/jason/p2/runs_gen_fc_3/public_128_100_32_1_inf_1e-06_0.0_0.01_5e-05_0.0_64_3_500000_LeakyReLU_0.0"
    # run_id = "/home/jason/p2/runs_gen_fc_3/public_256_100_32_1_inf_1e-06_0.0_0.01_5e-05_0.0_64_3_500000_LeakyReLU_0.0"

    run_id = run_id.split("/")[-1]
    run_fp = os.path.join('runs_gen_fc_3/', run_id)
    args = parse_run_id(run_id)

    for i in range(400000, 500000 + 1, 10000):
        gen_fp = os.path.join(run_fp, 'netG_{}.pt'.format(i))
        if os.path.exists(gen_fp):
            print("Loading {}".format(gen_fp))

            G = Generator_FC([128], args.nz).to(device)
            G.load_state_dict(torch.load(gen_fp))
            G.eval()
    
            # Generate 2048 fake images
            noise = torch.randn(2048, 100).to(device)
            fake = G(noise)
            fake = fake.view(fake.size(0), 1, 28, 28)

            # Calculate Inception Score
            IS = get_IS(fake)
            print("Inception Score:", IS)

            # Calculate Frechet Inception Distance
            # FID = get_FID(fake)
            # print("Frechet Inception Distance:", FID)

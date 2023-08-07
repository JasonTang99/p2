import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

from inception_score_pytorch.inception_score import inception_score
from data import load_MNIST

def _get_MNIST_metrics(fp="data/MNIST/stats_private.npz"):
    """Since pytorch-fid requires a folder of images, we temporarily save the
    MNIST dataset to disk and then delete it.
    """
    # Check if stats were already calculated
    if os.path.exists(fp):
        print("MNIST stats already exist. Skipping...")
        return
    
    # Load MNIST dataset
    labeling_loader, public_loader, private_loader, test_loader = load_MNIST(64)
    
    # Calculate only for Private set
    dataset = private_loader.dataset

    # Create a temporary folder to store the images
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Save images to disk
    for i, (img, _) in enumerate(dataset):
        torchvision.utils.save_image(img, f"{tmp_dir}/{i}.png")

    # Calculate FID
    os.system(f"python -m pytorch_fid --save-stats {tmp_dir} {fp}")

    # Delete temporary folder
    os.system(f"rm -rf {tmp_dir}")


def get_FID(imgs, fp="data/MNIST/stats_private.npz"):
    """Wrapper for FID calculation
    imgs: Torch dataset of (3xHxW) numpy images in range [0, 1] (minimum 2048 images)
    """
    # Save images to disk
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    for i, img in enumerate(imgs):
        torchvision.utils.save_image(img, f"{tmp_dir}/{i}.png")

    # Calculate FID
    print(f"python -m pytorch_fid --device cuda:0 {tmp_dir} {fp}")
    fid = os.popen(f"python -m pytorch_fid --device cuda:0 {tmp_dir} {fp}").read()
    print(fid)
    fid = float(fid.split("FID: ")[-1])

    # Delete temporary folder
    os.system(f"rm -rf {tmp_dir}")
    return fid


def get_IS(imgs):
    """Wrapper for Inception Score calculation
    imgs: Torch dataset of (3xHxW) numpy images in range [0, 1]
    """
    # normalized to the range [-1, 1]
    imgs = (imgs - 0.5) * 2

    # Convert to 3 channels if necessary
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)

    # Calculate Inception Score
    IS, _ = inception_score(imgs, cuda=True, batch_size=32, resize=True)
    return IS


if __name__ == '__main__':
    _get_MNIST_metrics()

    labeling_loader, public_loader, private_loader, test_loader = load_MNIST(64)

    # Take 2048 images from the public set
    public_set = public_loader.dataset
    public_sample = torch.stack([public_set[i][0] for i in range(2048)])
    fid = get_FID(public_sample, fp="data/MNIST/stats_private.npz")
    # 141.14

    # Take 2048 images from the private set
    private_set = private_loader.dataset
    private_sample = torch.stack([private_set[i][0] for i in range(2048)])
    fid = get_FID(private_sample, fp="data/MNIST/stats_private.npz")
    # 62.43
    
    # Take 2048 images from the labeling set (both odd and even)
    labeling_set = labeling_loader.dataset
    labeling_sample = torch.stack([labeling_set[i][0] for i in range(2048)])
    fid = get_FID(labeling_sample, fp="data/MNIST/stats_private.npz")
    # 14.91

    # Take 2048 images from the test set
    test_set = test_loader.dataset
    test_sample = torch.stack([test_set[i][0] for i in range(2048)])
    fid = get_FID(test_sample, fp="data/MNIST/stats_private.npz")
    # 17.14

    exit(0)


    # Generate random images
    imgs = torch.rand(2048, 1, 28, 28)

    # Calculate FID
    fid = get_FID(imgs)
    print("FID:", fid)

    # Calculate Inception Score
    IS = get_IS(imgs)
    print(IS)

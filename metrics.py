import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

from inception_score_pytorch.inception_score import inception_score


MNIST_stats_fp = "data/MNIST/stats.npz"

def _get_MNIST_metrics():
    """Since pytorch-fid requires a folder of images, we temporarily save the
    MNIST dataset to disk and then delete it.
    """
    # Check if stats were already calculated
    if os.path.exists(MNIST_stats_fp):
        print("MNIST stats already exist. Skipping...")
        return
    
    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    # Concatenate train and test set
    dataset = torch.utils.data.ConcatDataset([trainset, testset])
    print("Dataset size:", len(dataset))

    # Create a temporary folder to store the images
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Save images to disk
    for i, (img, _) in enumerate(dataset):
        torchvision.utils.save_image(img, f"{tmp_dir}/{i}.png")

    # Calculate FID
    os.system(f"python -m pytorch_fid --save-stats {tmp_dir} {MNIST_stats_fp}")

    # Delete temporary folder
    os.system(f"rm -rf {tmp_dir}")


def get_FID(imgs):
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
    fid = os.popen(f"python -m pytorch_fid --device cuda:0 {tmp_dir} {MNIST_stats_fp}").read()
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

    # Generate random images
    imgs = torch.rand(2048, 1, 28, 28)

    # Calculate FID
    fid = get_FID(imgs)
    print("FID:", fid)

    # Calculate Inception Score
    IS = get_IS(imgs)
    print(IS)

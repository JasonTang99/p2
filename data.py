import torch
import torchvision
import torchvision.transforms as transforms

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_MNIST(batch_size):
    """Load MNIST dataset
    Train set of 60000 images. Split into 3 parts:
    - 20000 images for training a labeling model (since generated images don't have labels)
        - 2000 images per class
    - 40000 images for training a classifier
        - Odd numbers (20000 images) as a "public" training set
        - Even numbers (20000 images) as a "private" training set

    Test set of 10000 images. Used for evaluating the downstream classifier.
    """
    MNIST_fp = "data/MNIST"
    transform = transforms.ToTensor()

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Check if we need to split
    if os.path.exists("labeling_set.pt") and os.path.exists("public_set.pt") and os.path.exists("private_set.pt"):
        # Load from disk
        print("Loading MNIST splits from disk")
        labeling_set = torch.load(f"{MNIST_fp}/labeling_set.pt")
        public_set = torch.load(f"{MNIST_fp}/public_set.pt")
        private_set = torch.load(f"{MNIST_fp}/private_set.pt")
    else:
        # Split train set into 3 parts
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        generator1 = torch.Generator().manual_seed(42)
        labeling_set, train_set = torch.utils.data.random_split(train_set, [20000, 40000], generator1)
        
        # Separate into classes
        class_idxs = [[] for _ in range(10)]
        for i, (_, label) in enumerate(train_set):
            class_idxs[label].append(i)
        
        # Create public (odd) and private (even) sets
        public_idxs = []
        private_idxs = []
        for i in range(5):
            public_idxs += class_idxs[2*i+1]
            private_idxs += class_idxs[2*i]
        public_set = torch.utils.data.Subset(train_set, public_idxs)
        private_set = torch.utils.data.Subset(train_set, private_idxs)

        # Save so we can use the same split later
        torch.save(labeling_set, f"{MNIST_fp}/labeling_set.pt")
        torch.save(public_set, f"{MNIST_fp}/public_set.pt")
        torch.save(private_set, f"{MNIST_fp}/private_set.pt")

    # Create data loaders
    labeling_loader = torch.utils.data.DataLoader(labeling_set, batch_size=batch_size, shuffle=True)
    public_loader = torch.utils.data.DataLoader(public_set, batch_size=batch_size, shuffle=True)
    private_loader = torch.utils.data.DataLoader(private_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return labeling_loader, public_loader, private_loader, test_loader

# Load latent space dataset
def load_latent(batch_size, data_fp="data/wgan_latent_dataset.pt"):
    """Load latent space dataset
    """
    dataset = torch.load(data_fp)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


if __name__ == "__main__":
    # Test latent space
    # loader = load_latent(64)
    # print("Latent space dataset size:", len(loader.dataset))
    # print("Latent space sample shape:", next(iter(loader))[0].shape)

    # from models import Discriminator_FC, Generator_FC
    # D = Discriminator_FC(hidden_sizes=[16, 16], input_size=100).to(device)
    # G = Generator_FC(nz=32, hidden_sizes=[16, 32], output_size=100).to(device)

    # for i, data in enumerate(loader):
    #     print(data[0].shape)
    #     print(D(data[0]).shape)
    #     noise = torch.randn(64, 32).to(device)
    #     print(G(noise).shape)
    #     break

    labeling_loader, public_loader, private_loader, test_loader = load_MNIST(64)
    for _ in range(2):
        for images, labels in public_loader:
            print(images.shape, labels.shape)
        

    exit(0)


    # Test MNIST

    # Print sizes of each set
    print("Labeling set size:", len(labeling_loader.dataset))
    print("Public set size:", len(public_loader.dataset))
    print("Private set size:", len(private_loader.dataset))
    print("Test set size:", len(test_loader.dataset))

    # Print unique labels in each set
    print("Labeling set labels:", set([label for _, label in labeling_loader.dataset]))
    print("Public set labels:", set([label for _, label in public_loader.dataset]))
    print("Private set labels:", set([label for _, label in private_loader.dataset]))
    print("Test set labels:", set([label for _, label in test_loader.dataset]))

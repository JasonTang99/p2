import torch
import torchvision
import torchvision.transforms as transforms


def MNIST(batch_size):
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


if __name__ == "__main__":
    # Test MNIST
    train_loader, test_loader = MNIST(4)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(inputs.shape, labels.shape)
        # print min and max of inputs
        print(inputs.min(), inputs.max())
        break

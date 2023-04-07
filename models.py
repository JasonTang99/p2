import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class Discriminator(nn.Module):
    """ Fully-connected Discriminator
    Inputs:
    - hidden_sizes: list of hidden layer sizes
    - input_size: size of the input vector
    """
    def __init__(self, hidden_sizes=[64, 16], input_size=28**2):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0], bias=False),
            nn.ReLU(),
            *list(chain.from_iterable([[
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=False), 
                nn.ReLU()
            ] for i in range(len(hidden_sizes) - 1)])),
            nn.Linear(hidden_sizes[-1], 1, bias=False),
        )

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, self.input_size)
        return self.model(x)

class Weight_Clipper(object):
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data = module.weight.data.clamp(-self.clip_value, self.clip_value)


class Generator_MNIST(nn.Module):
    """ Convolutional Generator for MNIST image size (1, 28, 28)
    Inputs:
    - nz: size of the latent z vector
    - ngf: size of feature maps in generator
    - nc: number of channels in the output image
    """
    def __init__(self, nz=100, ngf=32, nc=1):
        super(Generator_MNIST, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf, nc, kernel_size=5, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 28 x 28
        )

    def forward(self, x):
        return self.model(x)


class Generator_FC(nn.Module):
    """ Fully-connected Generator
    Inputs:
    - nz: size of the latent z vector
    - hidden_sizes: list of hidden layer sizes
    - output_size: size of the output vector
    """
    def __init__(self, nz=100, hidden_sizes=[64, 16], output_size=28**2):
        super(Generator_FC, self).__init__()
        self.nz = nz
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(nz, hidden_sizes[0], bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            *list(chain.from_iterable([[
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=False),
                nn.ReLU(), 
                nn.BatchNorm1d(hidden_sizes[i])
            ] for i in range(1, len(hidden_sizes))])),
            nn.Linear(hidden_sizes[-1], output_size, bias=False),
        )

    def forward(self, x):
        return self.model(x)

# Setup Generator Weight Initialization
def G_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    # Test Discriminator
    D = Discriminator(hidden_sizes=[64, 16, 16, 16], input_size=28**2)
    x = torch.randn(1, 28**2)
    print(D(x))
    print(D.model)

    # Print max and min weights
    print("Max weight:", D.model[0].weight.data.max())
    print("Min weight:", D.model[0].weight.data.min())

    # Clip weights
    clipper = Weight_Clipper(0.01)
    D.apply(clipper)

    # Print max and min weights
    print("Max weight:", D.model[0].weight.data.max())
    print("Min weight:", D.model[0].weight.data.min())

    # Validate
    from opacus.validators import ModuleValidator
    errors = ModuleValidator.validate(D, strict=False)
    print("Errors:", errors)

    # Test Generator
    G = Generator_MNIST()
    G.apply(G_weights_init)
    z = torch.randn(16, 100, 1, 1)
    print(G(z).shape)
    print(G.model)

    # Test Generator_FC
    G_fc = Generator_FC(nz=100, hidden_sizes=[64, 16, 16], output_size=28**2)
    z = torch.randn(16, 100)
    print(G_fc(z).shape)
    print(G_fc.model)

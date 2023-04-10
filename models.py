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
    def __init__(self, hidden_sizes=[64, 16], input_size=28**2, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0], bias=False),
            activation,
            *list(chain.from_iterable([[
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=False), 
                activation
            ] for i in range(len(hidden_sizes) - 1)])),
            nn.Linear(hidden_sizes[-1], 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, self.input_size)
        return self.model(x)

# Convolutional Discriminator for MNIST image size (1, 28, 28)
class Discriminator_MNIST(nn.Module):
    def __init__(self, ndf=32, nc=1):
        super(Discriminator_MNIST, self).__init__()
        # 6 layer discriminator
        self.model = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 2, 4, 2, 1),
            nn.Flatten(),
            nn.Linear(ndf * 2, 1),
        )

    def forward(self, x):
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
            nn.ReLU(True),
            nn.BatchNorm1d(hidden_sizes[0]),
            *list(chain.from_iterable([[
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=False),
                nn.ReLU(True), 
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


# Autoencoder
class Encoder(nn.Module):
    def __init__(self, latent_size=32):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Flatten(),
            # Linear layers
            nn.Linear(32*3*3, 96),
            nn.ReLU(True),
            nn.Linear(96, self.latent_size),
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_size=32):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        
        self.model = nn.Sequential(
            # Linear layers
            nn.Linear(self.latent_size, 96),
            nn.ReLU(True),
            nn.Linear(96, 32*3*3),
            nn.ReLU(True),
            # Convolutional layers
            nn.Unflatten(1, (32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # test discriminator mnist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    D = Discriminator_MNIST(ndf=32, nc=1).to(device)
    x = torch.randn(16, 1, 28, 28).to(device)
    print(D(x).shape)

    from privacy import compute_empirical_bounds

    # compute empirical bounds
    c_g = compute_empirical_bounds(D, 0.01, (1, 1, 28, 28), ) 
    print(c_g)

    exit(0)


    # Test Encoder
    E = Encoder()

    # print a convolutional layer
    print(E.model[0].bias.shape)
    exit(0)

    x = torch.randn(32, 1, 28, 28)
    print(E(x).shape)

    # Test Decoder
    D = Decoder()
    xp = D(E(x))
    print(xp.shape)

    assert x.shape == xp.shape

    


    # Test Discriminator
    D = Discriminator(hidden_sizes=[64, 16, 16, 16], input_size=28**2)
    x = torch.randn(1, 28**2)
    print(D(x))
    print(D.model)

    D_tanh = Discriminator(hidden_sizes=[64, 16, 16, 16], input_size=28**2, activation=nn.Tanh())
    x = torch.randn(1, 28**2)
    print(D_tanh(x))
    print(D_tanh.model)

    # Print max and min weights
    print("Max weight:", D.model[0].weight.data.max())
    print("Min weight:", D.model[0].weight.data.min())

    # Clip weights
    clipper = Weight_Clipper(0.01)
    D.apply(clipper)

    # Print max and min weights
    print("Max weight:", D.model[0].weight.data.max())
    print("Min weight:", D.model[0].weight.data.min())

    exit()

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

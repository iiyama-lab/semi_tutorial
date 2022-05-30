import torch.nn as nn
import torch

#from dataloader import show_images


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                z_dim, 1024, kernel_size=4, stride=1, padding=0)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                1024, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.last = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 3, kernel_size=3, padding=1)),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.last = nn.Conv2d(256, 1, kernel_size=3, stride=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out


if __name__ == "__main__":
    G = Generator(z_dim=100)

    input_z = torch.randn(1, 100)
    input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

    fake_images = G(input_z)
    print(fake_images.shape)
    # show_images(fake_images.detach())
    #print (G)

    D = Discriminator()
    ret = D(fake_images)
    print(nn.Sigmoid()(ret))
    print("done")

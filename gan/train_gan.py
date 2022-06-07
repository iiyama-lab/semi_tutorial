#from tqdm.notebook import tqdm
# jupyter notebookで動かさない場合は下の方を有効にする
from tqdm import tqdm
import torch
import torch.nn as nn

from dataloader import GANImageDataset, ImageTransform, show_images
from model import Generator, Discriminator

class Train_model:
    def __init__(self, device):
        self.device = device

    def initialize(self, G, D, z_dim, setting):
        self.D = D
        self.G = G
        self.z_dim = z_dim
        g_lr = setting["g_lr"]
        d_lr = setting["d_lr"]
        beta1 = setting["beta1"]
        beta2 = setting["beta2"]
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), g_lr, [beta1, beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), d_lr, [beta1, beta2])
        # self.g_optimizer = torch.optim.SGD(self.G.parameters(), g_lr)
        # self.d_optimizer = torch.optim.SGD(self.D.parameters(), d_lr)

        self.D.to(self.device)
        self.G.to(self.device)

        torch.backends.cudnn.benchmark = True

        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def train(self, dataloader, num_epochs):
        num_images = len(dataloader.dataset)
        self.batch_size = dataloader.batch_size
        z = torch.randn(self.batch_size, self.z_dim, 1, 1).to(self.device)

        pbar_epoch = tqdm(total=num_epochs)
        pbar_epoch.set_description("epoch")
        for epoch in range(num_epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            self.G.train()
            self.D.train()
            pbar_batch = tqdm(total=num_images, leave=False)
            for i, images in enumerate(dataloader):
                _batch_size = images.size()[0]
                epoch_d_loss += self.train_D(images, _batch_size)
                epoch_g_loss += self.train_G(_batch_size)
                pbar_batch.set_postfix(
                    {"dLoss": epoch_d_loss / (i + 1), "gLoss": epoch_g_loss / (i + 1)}
                )
                pbar_batch.update(_batch_size)
            pbar_epoch.set_postfix(
                {"dLoss": epoch_d_loss / (i + 1), "gLoss": epoch_g_loss / (i + 1)}
            )
            pbar_epoch.update()

            sample, pred = self.generate_fake_images(z)
            show_images(sample)
            if epoch % 10 == 0:
                show_images(sample, f"out{epoch:04d}.png")

    def train_D(self, images, batch_size):
        if batch_size == 1:
            return 0
        self.D.zero_grad()

        images = images.to(self.device)
        label_real = torch.full((batch_size,), 0.0).to(self.device)
        label_fake = torch.full((batch_size,), 1.0).to(self.device)

        d_out_real = self.D(images)

        input_z = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
        fake_images = self.G(input_z)
        d_out_fake = self.D(fake_images)

        d_loss_real = self.criterion(d_out_real.view(-1), label_real)
        d_loss_fake = self.criterion(d_out_fake.view(-1), label_fake)
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item()

    def train_G(self, batch_size):
        if batch_size == 1:
            return 0
        self.G.zero_grad()

        label_real = torch.full((batch_size,), 0.0).to(self.device)

        input_z = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
        fake_images = self.G(input_z)
        d_out_fake = self.D(fake_images)

        g_loss = self.criterion(d_out_fake.view(-1), label_real)

        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item()

    def generate_fake_images(self, z):
        self.D.eval()
        self.G.eval()

        fake_images = self.G(z)
        d_out = self.D(fake_images).view(-1).to("cpu").detach().numpy()
        fake_images = fake_images.to("cpu").detach()
        return fake_images, d_out


if __name__ == "__main__":
    batch_size = 64
    z_dim = 100
    setting = {"g_lr": 1.0e-4, "d_lr": 5.0e-4, "beta1": 0.5, "beta2": 0.999}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = GANImageDataset("/root/data/share/face/train", transform=ImageTransform())
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    G = Generator(z_dim=z_dim)
    D = Discriminator()

    trainer = Train_model(device)
    trainer.initialize(G, D, z_dim, setting)
    trainer.train(dataloader, num_epochs=200)

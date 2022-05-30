# データローダを自作しましょう

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import os
import glob

import matplotlib.pyplot as plt
import numpy as np


class GANImageDataset(Dataset):
    """GAN用のImageDataset

    Attributes:
        filenames (list): 画像(PNG画像)のファイル名リスト
        transform (obj): 画像変換用の関数
    """

    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir: 画像が置いてあるディレクトリ名
            transform: 画像変換用の関数
        """
        self.transform = transform
        self.filenames = glob.glob(os.path.join(img_dir, "*/*.png"))
        print(f"{self.__len__()} images for training")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image


class ImageTransform():
    def __init__(self,  mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


def tensor2image(image,  mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Args:
        image: pytorch Tensor
    """
    inp = image.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def show_images(images, filename=None, ncols=8, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    nImages = images.shape[0]
    width = images.shape[3]
    height = images.shape[2]
    nrows = nImages // ncols

    buf = np.zeros((ncols*height, nrows*width, 3))
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx >= nImages:
                continue
            buf[c*height:(c+1)*height, r*width:(r+1)*width,
                :] = tensor2image(images[idx], mean, std)
            idx += 1

    fig, ax = plt.subplots()
    ax.imshow(buf)
    if filename is None:
        filename = "out.png"
    fig.savefig(filename)
    plt.close()


if __name__ == "__main__":
    real_dataset = GANImageDataset(
        "/root/data/share/face/train", transform=ImageTransform())
    img = real_dataset.__getitem__(20)

    fig, ax = plt.subplots()
    ax.imshow(tensor2image(img))
    fig.savefig("out.png")

    print("done")

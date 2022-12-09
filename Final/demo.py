import datetime
import platform
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm

import AutoEncoder


def shape_after_conv2d(width: int,
                       height: int,
                       kernel_size: Union[int, Tuple[int, int]],
                       padding: Union[int, Tuple[int, int]] = (0, 0),
                       stride: Union[int, Tuple[int, int]] = (1, 1),
                       dilation: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    """
    Calculation formula from:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    :param dilation:
    :param height:
    :param width:
    :param kernel_size:
    :param padding:
    :param stride:
    :return:
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    return (height + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) // stride[0] + 1, \
           (width + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) // stride[1] + 1


def plot_ae_outputs(encoder, decoder, n=10, save=False, name=""):
    plt.figure(figsize=(16, 4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    if not save:
        plt.show()
    else:
        plt.savefig(name, dpi=300)


if __name__ == "__main__":
    transformer = transforms.Compose([
        transforms.ToTensor(),
    ])
    batch_size = 256
    eta = 1e-3
    train_dataset = dset.FashionMNIST(root='./data', train=True, download=True, transform=transformer)
    test_dataset = dset.FashionMNIST(root="./data", train=False, download=True, transform=transformer)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    loss_fn = nn.MSELoss()

    dim = 32
    # Calculate the input size for the linear layers
    # [(W−K+2P)/S]+1
    # 28 - 7 + 2 * 0 / 1 + 1 = 22
    # 22 - 3 + 2 * 0 / 1 + 1 = 20
    picture_shape = (28, 28)
    linear_input_size = shape_after_conv2d(*shape_after_conv2d(*picture_shape, 7), 3)
    assert linear_input_size == (20, 20)
    encoder = AutoEncoder.ImgEncoder(dim, linear_input_size)
    decoder = AutoEncoder.ImgDecoder(dim, linear_input_size)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=eta)
    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    device = torch.device(device)
    # Train phase
    encoder.to(device)
    decoder.to(device)
    train_loss = []
    for epoch in range(epochs):
        ep_loss = []
        for img_bat, _ in tqdm(train_loader):
            img_bat = img_bat.to(device)
            encoded = encoder(img_bat)
            decoded = decoder(encoded)
            loss = loss_fn(decoded, img_bat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.detach().cpu().numpy())
        train_loss.append(np.mean(ep_loss))
        print(f"Epoch: {epoch}, loss: {np.mean(ep_loss)}")
    torch.save(encoder, f"{epochs}Epochs_MnistFashion_encoder_full.pth")
    torch.save(decoder, f"{epochs}Epochs_MnistFashion_decoder_full.pth")
    plot_ae_outputs(encoder, decoder)

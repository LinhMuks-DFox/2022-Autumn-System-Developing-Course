import AutoEncoder
import torch
import matplotlib.pyplot as plt
import torch.utils.data as data
import numpy as np
from typing import *
from tqdm import tqdm


class UtilityFunctions:
    @staticmethod
    def plot_auto_encoder_outputs(encoder_model,
                                  decoder_model,
                                  test_dataset: data.DataLoader,
                                  device: torch.device,
                                  n=10,
                                  save=False,
                                  name="",
                                  ):
        plt.figure(figsize=(16, 4.5))
        targets = test_dataset.targets.numpy()
        t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
            encoder_model.eval()
            decoder_model.eval()
            with torch.no_grad():
                rec_img = decoder_model(encoder_model(img))
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

    @staticmethod
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


class AutoEncoderTrainer:

    def __init__(self,
                 img_size: Tuple[int, int, int],
                 ndim: int,
                 loss_function,
                 optimizer,
                 epochs,
                 train_data_loader,
                 test_data_loader,
                 device,
                 learning_rata=1e-3) -> None:
        self.kernel_size = (31, 31)
        self.linear_input_size_of_auto_encoder = UtilityFunctions.shape_after_conv2d(
            *UtilityFunctions.shape_after_conv2d(*img_size[1:], 31), 31)
        print("Training Encoder, Decoder with parameter<linear_input_size_of_auto_encoder>=",
              self.linear_input_size_of_auto_encoder)
        self.encoder = AutoEncoder.ImgEncoder(ndim, self.linear_input_size_of_auto_encoder,
                                              kernel_size=self.kernel_size, img_channels=img_size[0]).to(device)
        self.decoder = AutoEncoder.ImgDecoder(ndim, self.linear_input_size_of_auto_encoder,
                                              img_channels=1, kernel_size=self.kernel_size).to(device)
        self.img_size_: Tuple[int, int, int]
        self.loss_function = loss_function
        opt_param = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optimizer(opt_param, lr=learning_rata)
        self.epochs = epochs
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.device = device
        self.train_phase_loss = []
        self.test_phase_loss = []

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def train(self):
        print("Training AutoEncoder")
        self.encoder.train()
        self.decoder.train()
        for epoch in range(self.epochs):
            epoch_loss = []
            for img_bat, _ in tqdm(self.train_data_loader):
                img_bat = img_bat.to(self.device)
                encoded = self.encoder(img_bat)
                decoded = self.decoder(encoded)
                loss = self.loss_function(decoded, img_bat)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.detach().cpu().numpy())
            self.train_phase_loss.append((l := np.mean(epoch_loss)))
            print(f"Epoch {epoch + 1} loss: {l}")
        return self

    def test(self):
        print("Testing AutoEncoder")
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            for epoch in range(self.epochs):
                ep_loss = []
                for img_bat, _ in tqdm(self.test_data_loader):
                    img_bat = img_bat.to(self.device)
                    encoded = self.encoder(img_bat)
                    decoded = self.decoder(encoded)
                    loss = self.loss_function(decoded, img_bat)
                    ep_loss.append(loss.detach().cpu().numpy())
                self.test_phase_loss.append((l := np.mean(ep_loss)))
                print(f"Test Epoch:{epoch + 1}, loss{l}\n")
        return self

    def plot_loss(self):
        plt.plot(self.train_phase_loss, label='train')
        plt.plot(self.test_phase_loss, label='test')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        return self

    def plot_reconstructed_images(self, test_dataset, n=10, save=False, name="reconstructed_images.png"):
        UtilityFunctions.plot_auto_encoder_outputs(self.encoder,
                                                   self.decoder,
                                                   test_dataset,
                                                   n=n, save=save, name=name, device=self.device)
        return self

    def save(self, name):
        torch.save(self.encoder.state_dict(), name + '_encoder.pth')
        torch.save(self.decoder.state_dict(), name + '_decoder.pth')
        return self

    def load(self, name):
        self.encoder.load_state_dict(torch.load(name + '_encoder.pth'))
        self.decoder.load_state_dict(torch.load(name + '_decoder.pth'))

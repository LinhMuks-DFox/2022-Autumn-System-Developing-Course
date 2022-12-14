import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Trainer import *

batch_size = 256
learning_rate = 1e-3
train_data = dset.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = dset.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
img_size = (1, 28, 28)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainer = AutoEncoderTrainer(
    img_size=img_size,
    ndim=32,
    loss_function=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam,
    epochs=50,
    train_data_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True),
    device=device,
    test_data_loader=DataLoader(test_data, batch_size=batch_size, shuffle=False),
).train().test().plot_loss().plot_reconstructed_images(test_data).save("auto_encoder")

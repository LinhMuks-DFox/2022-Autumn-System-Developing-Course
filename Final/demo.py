import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Trainer import *

batch_size = 1
learning_rate = 1e-3
# train_data = dset.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# test_data = dset.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
data_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((500, 500)),
])
train_data = dset.Flowers102(root='./data', split="train", transform=data_transformer, download=True)
test_data = dset.Flowers102(root='./data', split="test", transform=data_transformer, download=True)
data, label = train_data[0]
print(data.shape)
print(label)
img_size = (3, 500, 500)
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

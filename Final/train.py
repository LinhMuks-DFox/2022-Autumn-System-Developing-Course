import random
from typing import Tuple, Any

import numpy as np
import torch.utils.data
import torchvision.transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

from ImageAutoEncoder.ImageAutoEncoder import *
from mklearn.model_train.model_training import *

data_transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
])
demo_dataset = MNIST(root='data/', download=True, transform=data_transformer)
train_dataloader = torch.utils.data.DataLoader(demo_dataset, batch_size=50, shuffle=True)
test_loader = torch.utils.data.DataLoader(demo_dataset, batch_size=50, shuffle=False)
data = "./data"


def train_kernel(model: Any, data, label) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = model.encoder_(data)
    decoded = model.decoder_(encoded)
    return decoded, data


torch.manual_seed(666)

model = ImageAutoEncoder(np.array([1, 28, 28]), 32)

alchemy_furnace = AlchemyFurnace(parameter := AlchemyParameters(
    model=model,
    model_name="ImageAutoEncoder",
    optimizer=torch.optim.Adam(model.parameters()),
    loss_function=torch.nn.MSELoss(),
    epochs=20,
    device="cuda:0",
    train_set=demo_dataset,
    train_kernel=train_kernel,
    train_data_loader=train_dataloader,
    test_data_loader=test_loader,
    test_in_train=False,
    test_kernel=train_kernel,
    verbose=True,
    validation_loader=test_loader
))
alchemy_furnace.train()
alchemy_furnace.score()
alchemy_furnace.plot_loss()
alchemy_furnace.save()
print("Overall, test loss: ", alchemy_furnace.test_loss_)
# plots
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i in range(5):
    r = random.randint(100, 1000)
    org_data, label = demo_dataset[i + r]
    org_data = org_data.unsqueeze(0).to(torch.device("cuda:0"))
    encoded = model.encoder_(org_data)
    decoded = model.decoder_(encoded)
    axes[0, i].imshow(org_data.detach().cpu().numpy()[0, 0, :, :])
    axes[0, i].set_title(f"Original label: {label}")
    axes[1, i].imshow(decoded.detach().cpu().numpy()[0, 0, :, :])
    axes[1, i].set_title(f"AutoEncoder Output: {label}")

plt.savefig("ImageAutoEncoder-MNIST-IO-PNG.png", dpi=300)

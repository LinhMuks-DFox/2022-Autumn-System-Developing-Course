from typing import Tuple, Any

import numpy as np
import torch.utils.data
import torchvision.transforms
from torchvision.datasets import MNIST

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
    epochs=3,
    device="cuda:0",
    train_set=demo_dataset,
    train_kernel=train_kernel,
    train_data_loader=train_dataloader,
    test_data_loader=test_loader,
    test_in_train=True,
    test_kernel=train_kernel,
    verbose=True
))
alchemy_furnace.train()
alchemy_furnace.plot_loss()
# print(parameter)

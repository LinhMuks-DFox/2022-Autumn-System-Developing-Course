__all__ = [
    "AlchemyParameters",
    "AlchemyFurnace",
]

import dataclasses
import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
import torch.utils.data.dataset
import tqdm

from mklearn.core.learn_module import LearnModule

ParameterType = TypeVar("ParameterType", bound="AbstractParameter")
ComputableKernelFunctionType = Callable[[Callable, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


@dataclasses.dataclass(frozen=True)
class AlchemyParameters:
    model: Optional[Union[torch.nn.Module, LearnModule]] = None
    model_name: Optional[str] = ""
    optimizer: Optional[torch.optim.Optimizer] = None
    loss_function: Optional[torch.nn.Module] = None
    device: Optional[Union[torch.device, str]] = None
    epochs: Optional[int] = 0
    save_path: Optional[str] = ""
    load_model: Optional[bool] = False
    save_model_when_destruct: Optional[bool] = False

    train_set: Optional[torch.utils.data.Dataset] = None
    test_set: Optional[torch.utils.data.Dataset] = None
    validation_set: Optional[torch.utils.data.Dataset] = None
    train_data_loader: Optional[torch.utils.data.DataLoader] = None
    test_data_loader: Optional[torch.utils.data.DataLoader] = None
    validation_loader: Optional[torch.utils.data.DataLoader] = None

    verbose: Optional[bool] = False
    test_in_train: Optional[bool] = False

    train_kernel: Optional[ComputableKernelFunctionType] = None
    test_kernel: Optional[ComputableKernelFunctionType] = None

    on_call: Optional[Callable] = None
    on_save: Optional[Callable] = None
    on_load: Optional[Callable] = None
    on_train: Optional[Callable] = None
    on_test: Optional[Callable] = None
    on_plot: Optional[Callable] = None
    on_init: Optional[Callable] = None
    on_finalize: Optional[Callable] = None
    on_forward: Optional[Callable] = None

    after_call: Optional[Callable] = None
    after_save: Optional[Callable] = None
    after_load: Optional[Callable] = None
    after_train: Optional[Callable] = None
    after_test: Optional[Callable] = None
    after_plot: Optional[Callable] = None
    after_init: Optional[Callable] = None
    after_finalize: Optional[Callable] = None
    after_forward: Optional[Callable] = None


class AlchemyFurnace:
    __slots__ = [
        "metadata_",
        "model_",
        "save_path",
        "test_kernel_",
        "test_loss_",
        "train_kernel_",
        "train_loss_"
    ]

    def __init__(self, metadata: AlchemyParameters):
        self.metadata_ = metadata

        if self.metadata_.verbose:
            print("Initializing Alchemy Furnace...")

        if self.metadata_.on_init is not None:
            self.metadata_.on_init(self)

        self.model_ = self.metadata_.model if self.metadata_.load_model is False else None
        if self.metadata_.load_model:
            self.load()

        self.move_model_to_device()
        if self.metadata_.verbose:
            print("model moved to device")

        self.test_loss_ = np.empty(self.metadata_.epochs)
        self.train_loss_ = np.empty(self.metadata_.epochs)

        if self.metadata_.train_kernel is not None:
            self.train_kernel_ = self.metadata_.train_kernel
        else:
            self.train_kernel_ = self._default_train_kernel
            if self.metadata_.verbose:
                print("Alchemy Furnace using default training kernel.")

        if self.metadata_.test_kernel is not None:
            self.test_kernel_ = self.metadata_.test_kernel
        else:
            self.test_kernel_ = self._default_train_kernel
            if self.metadata_.verbose:
                print("Alchemy Furnace using default testing kernel.")

        if self.metadata_.after_init is not None:
            self.metadata_.after_init(self)

        self.save_path = os.path.join(self.metadata_.save_path, f"{self.metadata_.model_name}.pt")
        if self.metadata_.verbose:
            print(f"Alchemy Furnace save path {self.save_path}")

    def move_model_to_device(self, device: Optional[Union[str, torch.device]] = None):
        if device is None and self.metadata_.device is None:
            raise ValueError("No device specified.")
        elif device is not None:
            self.model_.to(device if isinstance(device, torch.device) else torch.device(device))
        else:
            self.model_.to(self.metadata_.device
                           if isinstance(self.metadata_.device, torch.device)
                           else torch.device(self.metadata_.device))
        return self

    def train(self):
        if self.model_ is None:
            raise AttributeError("model undefined")

        if self.metadata_.train_data_loader is None:
            raise AttributeError("train_data_loader undefined")

        if self.metadata_.on_train is not None:
            self.metadata_.on_train(self)
        for epoch in range(self.metadata_.epochs):
            epoch_loss = []
            if self.metadata_.verbose:
                print(f"Train Epoch: {epoch}")
            for inputs, labels in tqdm.tqdm(self.metadata_.train_data_loader):
                inputs = inputs.to(self.metadata_.device)
                labels = labels.to(self.metadata_.device)
                self.metadata_.optimizer.zero_grad()

                outputs, labels = self.train_kernel_(self.model_, inputs, labels)

                loss = self.metadata_.loss_function(outputs, labels)
                loss.backward()
                self.metadata_.optimizer.step()
                epoch_loss.append(loss.detach().cpu().numpy())
            self.train_loss_[epoch] = np.mean(epoch_loss)
            if self.metadata_.verbose:
                print(f"In this epoch, the average loss is: {self.train_loss_[epoch]}")

            if self.metadata_.test_in_train:
                test_epoch_loss = []
                if self.metadata_.verbose:
                    print(f"Test in Training Epoch: {epoch}")
                for inputs, labels in tqdm.tqdm(self.metadata_.test_data_loader):
                    inputs = inputs.to(self.metadata_.device)
                    labels = labels.to(self.metadata_.device)
                    outputs, labels = self.test_kernel_(self.model_, inputs, labels)
                    loss = self.metadata_.loss_function(outputs, labels)
                    test_epoch_loss.append(loss.detach().cpu().numpy())
                self.test_loss_[epoch] = np.mean(test_epoch_loss)
                if self.metadata_.verbose:
                    print(f"In this epoch, the average test loss is: {self.test_loss_[epoch]}")

        if self.metadata_.after_train is not None:
            self.metadata_.after_train(self)
        return self

    @torch.no_grad()
    def score(self, validation_set: Optional[torch.utils.data.DataLoader] = None):
        if self.model_ is None:
            raise AttributeError("model undefined")
        if self.metadata_.test_data_loader is None:
            raise AttributeError("test_data_loader undefined")

        if self.metadata_.on_test is not None:
            self.metadata_.on_test(self)

        self.model_.to(self.metadata_.device)
        score_loader = validation_set if validation_set is not None else self.metadata_.validation_loader
        if score_loader is None:
            raise ValueError(
                "Score failed. validation_loader in self.metadata_ is none, and validation_set arg is none.")

        with torch.no_grad():
            test_batch_loss = []
            if self.metadata_.verbose:
                print(f"Testing/Validating\n")
            for data, label in tqdm.tqdm(score_loader):
                data = data.to(self.metadata_.device)
                label = label.to(self.metadata_.device)
                output, label = self.test_kernel_(self.model_, data, label)
                loss = self.metadata_.loss_function(output, label)
                test_batch_loss.append(loss.detach().cpu().numpy())
            self.test_loss_ = np.mean(test_batch_loss)
            if self.metadata_.verbose:
                print(f"In test/validate phase, the average loss is: {self.test_loss_}\n")
        if self.metadata_.after_test is not None:
            self.metadata_.after_test(self)
        return self

    def save(self, path: Optional[str] = None):
        if self.model_ is None:
            AttributeError("model is not None")

        if self.metadata_.save_path is None or len(self.metadata_.save_path) == 0:
            AttributeError("save_path is not None")

        if self.metadata_.on_save is not None:
            self.metadata_.on_save(self)
        des = path if path is not None else self.save_path
        torch.save(self.model_, des)

        if self.metadata_.after_save is not None:
            self.metadata_.after_save(self)

        return self

    def load(self):
        if self.model_ is not None:
            raise AttributeError("model defined!")
        if self.metadata_.save_path is None or len(self.metadata_.save_path) == 0:
            AttributeError("save_path is not empty/undefined")

        if self.metadata_.on_load is not None:
            self.metadata_.on_load(self)
        self.model_ = torch.load(self.save_path)
        if self.metadata_.after_load is not None:
            self.metadata_.after_load(self)

        return self

    def plot_loss(self, figure_name="Train Test Loss in Epoch",
                  dpi=200, save: bool = False, save_path: str = ""):
        if self.metadata_.on_plot is not None:
            self.metadata_.on_plot(self)
        plt.plot(self.train_loss_, label="train loss")
        if self.metadata_.test_in_train:
            plt.plot(self.test_loss_, label="test loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(figure_name)
        plt.legend()
        if save:
            plt.savefig(f".\\{figure_name}.png" if len(save_path) == 0 else save_path, dpi=dpi)
        plt.show()
        if self.metadata_.after_plot is not None:
            self.metadata_.after_plot(self)
        return self

    def forward(self, sample):
        if self.model_ is None:
            raise AttributeError("model undefined")

        if self.metadata_.on_forward is not None:
            self.metadata_.on_forward(self, sample)

        if self.model_ is None:
            raise AttributeError("model undefined")

        ret = self.model_(sample)

        if self.metadata_.after_forward is not None:
            self.metadata_.after_forward(self, sample)

        return ret

    predict = forward

    def __call__(self, sample):
        if self.metadata_.on_call is not None:
            self.metadata_.on_call(self, sample)
        ret = self.forward(sample)
        if self.metadata_.after_call is not None:
            self.metadata_.after_call(self, sample)
        return ret

    def __repr__(self):
        return f"AlchemyFurnace: {self.metadata_}"

    def __str__(self):
        return self.__repr__()

    def __del__(self):
        if self.metadata_.on_finalize is not None:
            self.metadata_.on_finalize(self)
        if self.metadata_.save_model_when_destruct:
            self.save()
        if self.metadata_.after_finalize is not None:
            self.metadata_.after_finalize(self)

    @staticmethod
    def _default_train_kernel(model, inputs, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        return model(inputs), labels

    fit = train

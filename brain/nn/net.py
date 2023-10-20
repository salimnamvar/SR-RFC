""" Neural Network Model

"""

# region Imported Dependencies
import importlib
import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from brain.nn.util.param import Param
from brain.util.cfg.config import BrainConfig


# endregion Imported Dependencies


class Net:
    def __init__(self, a_name: str, a_arch: Param, a_optim: Param, a_lrs: Param, a_loss: Param, a_device: torch.device,
                 a_half: bool) -> None:
        self.name: str = a_name
        self.arch_param: Param = a_arch
        self.optim_param: Param = a_optim
        self.lrs_param: Param = a_lrs
        self.loss_param: Param = a_loss
        self.device: torch.device = a_device
        self.half: bool = a_half

        # Initialize Architecture
        self.arch: nn.Module = None
        self.__init_arch()

        # Initialize Optimizer
        self.optim: optim = None
        self.__init_optim()

        # Initialize Learning Rate Scheduler
        self.lrs: optim.lr_scheduler = None
        self.__init_lrs()

        # Initialize Loss Function
        self.loss: callable = None
        self.__init_loss()

        # Logging
        self.cfg: BrainConfig = BrainConfig.get_instance()
        self.logger = logging.getLogger(self.cfg.log.name + '.' + self.name)

    def __init_arch(self) -> None:
        # Build Architecture
        try:
            module_name = f"brain.nn.{self.arch_param.name}.arch"
            module = importlib.import_module(module_name)
            arch_class = getattr(module, 'Arch')
        except (AttributeError, ModuleNotFoundError):
            raise ValueError('Invalid `arch.name` is entered.')
        self.arch: nn.Module = arch_class(**self.arch_param.kwargs)

        # Target Device
        self.arch.to(self.device)

        # Half Precision
        if self.half:
            self.arch.half()

    def __init_optim(self) -> None:
        try:
            optimizer = getattr(optim, self.optim_param.name, None)
        except ModuleNotFoundError:
            raise ValueError(f'Invalid `optim.name` is entered: {self.optim_param.name}')

        self.optim = optimizer(params=self.arch.parameters(), **self.optim_param.kwargs)

    def __init_lrs(self) -> None:
        try:
            lrs = getattr(optim.lr_scheduler, self.lrs_param.name, None)
        except ModuleNotFoundError:
            raise ValueError(f'Invalid `lrs.name` is entered: {self.lrs_param.name}')

        self.lrs = lrs(optimizer=self.optim, **self.lrs_param.kwargs)

    def __init_loss(self) -> None:
        try:
            loss = getattr(nn, self.loss_param.name, None)
        except ModuleNotFoundError:
            raise ValueError(f'Invalid `loss.name` is entered: {self.loss_param.name}')

        self.loss = loss(**self.loss_param.kwargs)

    def train(self, a_data_loader: DataLoader, a_writer: SummaryWriter, a_epoch: int) -> float:
        self.arch.train()
        epoch_loss: float = 0.0
        for i, (inputs, targets) in tqdm(enumerate(a_data_loader), desc=f"Epoch {a_epoch} - Mini-Batch Training "):
            inputs = [input_tensor.to(self.device, dtype=torch.long) for input_tensor in inputs]
            targets = targets.to(self.device, dtype=torch.float)
            outputs = self.arch(*inputs)
            self.optim.zero_grad()
            loss = self.loss(outputs, targets)
            self.optim.zero_grad()
            self.logger.info(f"Batch {i}'s Training Loss is {loss.item()}.")
            epoch_loss += loss.item() * targets.size(0)
            loss.backward()
            self.optim.step()
            a_writer.add_scalar('data/train_batch_loss', loss, i)
        epoch_loss /= len(a_data_loader)
        return epoch_loss

    def validate(self, a_data_loader: DataLoader, a_writer: SummaryWriter, a_epoch: int) -> float:
        self.arch.eval()
        epoch_loss: float = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in tqdm(enumerate(a_data_loader), desc=f"Epoch {a_epoch} - Mini-Batch Validation "):
                inputs = [input_tensor.to(self.device, dtype=torch.long) for input_tensor in inputs]
                targets = targets.to(self.device, dtype=torch.float)
                outputs = self.arch(*inputs)
                loss = self.loss(outputs, targets)
                self.logger.info(f"Batch {i}'s Validation Loss is {loss.item()}.")
                epoch_loss += loss.item() * targets.size(0)
                a_writer.add_scalar('data/val_batch_loss', loss, i)
            epoch_loss /= len(a_data_loader)
        self.lrs.step(epoch_loss)
        return epoch_loss

    def test(self, a_weights: str, a_data_loader: DataLoader) -> pd.DataFrame:
        # Load the trained weights
        weights = torch.load(a_weights)['state_dict']
        self.arch.load_state_dict(weights)

        results = pd.DataFrame(columns=['id', 'reactivity'])

        self.arch.eval()
        with torch.no_grad():
            for i, (inputs, (ids_min, ids_max)) in tqdm(enumerate(a_data_loader), desc='Testing is in process'):
                inputs = [input_tensor.to(self.device, dtype=torch.long) for input_tensor in inputs]
                outputs = self.arch(*inputs)

                for j in range(len(outputs)):
                    ids = np.arange(ids_min[j], ids_max[j])
                    df = pd.DataFrame({'id': ids, 'reactivity': outputs.cpu()[j][:len(ids)]})
                    results = pd.concat([results, df])
        results = results.set_index('id')
        return results


class Nets:
    def __init__(self) -> None:
        self._items: Dict[str] = {}

    def append(self, a_net: Net) -> None:
        self._items[a_net.name] = a_net

    def pop(self, a_key: str) -> None:
        self._items.pop(a_key)

    @property
    def items(self) -> dict:
        return self._items

    def __getitem__(self, a_key: str) -> Net:
        return self._items[a_key]

    def __len__(self) -> int:
        return len(self._items)

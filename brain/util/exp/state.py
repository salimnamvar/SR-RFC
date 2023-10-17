""" Experiment State

"""

# region Imported Dependencies
import os
import datetime
from dataclasses import dataclass
from typing import List

import torch

from brain.util.log.setup import SetupLogger


# endregion Imported Dependencies


@dataclass
class Epoch:
    id: int
    loss: float


class Epochs:
    def __init__(self) -> None:
        self._items: List[Epoch] = []

    def append(self, a_state: Epoch) -> None:
        self._items.append(a_state)

    @property
    def items(self) -> List[Epoch]:
        return self._items

    def __getitem__(self, a_index: int) -> Epoch:
        return self._items[a_index]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def min(self) -> float:
        return min([epoch.loss for epoch in self.items]) if len(self) > 0 else 0


class Experiment:
    def __init__(self, a_root_path: str, a_cfg_name: str, a_run: int):
        self.cfg_name: str = a_cfg_name
        self.run: int = a_run
        self.epochs: Epochs = Epochs()
        self.root_path: str = a_root_path
        self.time: datetime.datetime = datetime.datetime.now()
        self.id: int = len(os.listdir(self.root_path))
        self.path: str = os.path.join(self.root_path, 'EXP-{}_{}_{:%Y-%m-%d-%H-%M-%p}_Run-{}'.format(self.id,
                                                                                                     self.cfg_name,
                                                                                                     self.time,
                                                                                                     self.run))
        os.makedirs(self.path, exist_ok=True)
        SetupLogger(a_filename=os.path.join(self.path, 'exp.log'))

    def save_epoch(self, a_loss: float, a_epoch: int, a_state: dict):
        if a_loss >= self.epochs.min:
            checkpoint_name = "E-%03d_Loss-%0.5f.pth.tar" % (a_epoch, a_loss)
            torch.save(a_state, os.path.join(self.path, checkpoint_name))


class Experiments:
    def __init__(self, a_root_path: str) -> None:
        self.root_path: str = a_root_path
        self._items: List[Experiment] = []

    def append(self, a_cfg_name: str, a_run: int) -> None:
        self._items.append(Experiment(a_cfg_name=a_cfg_name, a_run=a_run, a_root_path=self.root_path))

    @property
    def items(self) -> List[Experiment]:
        return self._items

    def __getitem__(self, a_index: int) -> Experiment:
        return self._items[a_index]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def experiment(self) -> Experiment:
        return self._items[-1]
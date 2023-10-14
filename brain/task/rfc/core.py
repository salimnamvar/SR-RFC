""" Experiment-1


"""


# region Imported Dependencies
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from brain.nn.net import Net
from brain.nn.util.param import Param
from brain.task.base import BaseTask
from brain.util.cfg.config import BrainConfig
from brain.util.data.dataset import SRRFCDataset
from brain.util.data.load import load_indices
# endregion Imported Dependencies


class Task(BaseTask):
    def __init__(self):
        super().__init__()
        self.iteration: int = -1
        self.cfg: BrainConfig = BrainConfig.get_instance()
        self.model: Net = None
        self.__init_model()

    def __init_model(self):
        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arch = Param(name=self.cfg.nn.arch.name, kwargs=self.cfg.nn.arch.kwargs)
        optim = Param(name=self.cfg.optim.name, kwargs=self.cfg.optim.kwargs)
        lrs = Param(name=self.cfg.lrs.name, kwargs=self.cfg.lrs.kwargs)
        loss = Param(name=self.cfg.loss.name, kwargs=self.cfg.loss.kwargs)
        self.model = Net(a_arch=arch, a_optim=optim, a_lrs=lrs, a_loss=loss, a_device=device, a_half=self.cfg.nn.half)

    def __train_loader(self) -> Tuple[DataLoader, DataLoader]:
        dataset = SRRFCDataset(a_file=self.cfg.data.train, a_chunk_size=self.cfg.data.chunk_size)

        # Samplers
        train_idx, val_idx = load_indices(a_train_idx_file=self.cfg.data.train_idx,
                                          a_val_idx_file=self.cfg.data.val_idx)
        train_sampler = SubsetRandomSampler(indices=train_idx)
        val_sampler = SubsetRandomSampler(indices=val_idx)

        # Data loaders
        train_loader = DataLoader(dataset=dataset, batch_size=self.cfg.train.batch, sampler=train_sampler, shuffle=True)
        val_loader = DataLoader(dataset=dataset, batch_size=self.cfg.val.batch, sampler=val_sampler, shuffle=True)
        return train_loader, val_loader

    def train(self):
        train_loader, val_loader = self.__train_loader()

        for run in range(0, self.cfg.train.run):
            for ep in range(0, self.cfg.train.epoch):
                self.model.train(a_data_loader=train_loader)
                self.model.validate(a_data_loader=val_loader)

    def test(self):
        NotImplemented

    def split_data(self):
        dataset = SRRFCDataset(a_file=self.cfg.data.train, a_chunk_size=1000)
        num_data = len(dataset)
        indices = list(range(0, num_data - 1))
        np.random.shuffle(indices)
        split = int(np.floor(self.cfg.data.val_size * num_data))
        train_idx, valid_idx = indices[split:], indices[:split]
        with open(self.cfg.data.train_idx, "w") as train_file:
            for idx in train_idx:
                train_file.write(f"{idx}\n")

        with open(self.cfg.data.val_idx, "w") as valid_file:
            for idx in valid_idx:
                valid_file.write(f"{idx}\n")
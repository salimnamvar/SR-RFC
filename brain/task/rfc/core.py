""" Experiment-1


"""
import logging

# region Imported Dependencies
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from brain.nn.net import Net, Nets
from brain.nn.util.param import Param
from brain.task.base import BaseTask
from brain.util.cfg.config import BrainConfig
from brain.util.data.dataset import TrainDataset
from brain.util.data.load import load_indices, Loaders, Loader
from brain.util.exp.state import Experiments


# endregion Imported Dependencies


class Task(BaseTask):
    def __init__(self, a_name: str = 'RFC.TASK'):
        super().__init__()
        self.name: str = a_name
        self.iteration: int = -1
        self.cfg: BrainConfig = BrainConfig.get_instance()
        self.models: Nets = Nets()
        self.loaders: Loaders = Loaders()
        self.__init_model()
        self.exps: Experiments = Experiments(a_root_path=self.cfg.tsk.exp_dir)
        self.logger = logging.getLogger(self.cfg.log.name + '.' + self.name)

    def __init_model(self):
        # Shared Parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arch = Param(name=self.cfg.nn.arch.name, kwargs=self.cfg.nn.arch.kwargs)
        optim = Param(name=self.cfg.optim.name, kwargs=self.cfg.optim.kwargs)
        lrs = Param(name=self.cfg.lrs.name, kwargs=self.cfg.lrs.kwargs)
        loss = Param(name=self.cfg.loss.name, kwargs=self.cfg.loss.kwargs)

        if self.cfg.data.inc_exp_type:
            model = Net(a_name='SRRFC', a_arch=arch, a_optim=optim, a_lrs=lrs, a_loss=loss, a_device=device,
                        a_half=self.cfg.nn.half)
            self.models.append(model)

        else:
            # DMS Model
            model_dms = Net(a_name='DMS', a_arch=arch, a_optim=optim, a_lrs=lrs, a_loss=loss, a_device=device,
                            a_half=self.cfg.nn.half)
            self.models.append(model_dms)

            # 2A3 Model
            model_2a3 = Net(a_name='2A3', a_arch=arch, a_optim=optim, a_lrs=lrs, a_loss=loss, a_device=device,
                            a_half=self.cfg.nn.half)
            self.models.append(model_2a3)

    def __init_train_loader(self):
        # Dataset
        dataset = TrainDataset(a_file=self.cfg.data.train, a_chunk_size=self.cfg.data.chunk_size,
                               a_max_length=self.cfg.data.max_length, a_inc_exp_type=self.cfg.data.inc_exp_type)

        if self.cfg.data.inc_exp_type:
            # Data Loader
            train_idx, val_idx = load_indices(a_train_idx_file=self.cfg.data.train_idx,
                                              a_val_idx_file=self.cfg.data.val_idx)
            train_sampler = SubsetRandomSampler(indices=train_idx)
            val_sampler = SubsetRandomSampler(indices=val_idx)
            train_loader = DataLoader(dataset=dataset, batch_size=self.cfg.train.batch, sampler=train_sampler)
            val_loader = DataLoader(dataset=dataset, batch_size=self.cfg.val.batch, sampler=val_sampler)
            loader = Loader(name='SRRFC', train=train_loader, val=val_loader)
            self.loaders.append(loader)

        else:
            # DMS Data Loaders
            train_idx_dms, val_idx_dms = load_indices(a_train_idx_file=self.cfg.data.train_idx[0],
                                                      a_val_idx_file=self.cfg.data.val_idx[0])
            train_sampler_dms = SubsetRandomSampler(indices=train_idx_dms)
            val_sampler_dms = SubsetRandomSampler(indices=val_idx_dms)
            train_loader_dms = DataLoader(dataset=dataset, batch_size=self.cfg.train.batch, sampler=train_sampler_dms)
            val_loader_dms = DataLoader(dataset=dataset, batch_size=self.cfg.val.batch, sampler=val_sampler_dms)
            loader_dms = Loader(name='DMS', train=train_loader_dms, val=val_loader_dms)
            self.loaders.append(loader_dms)

            # 2A3 Data Loaders
            train_idx_2a3, val_idx_2a3 = load_indices(a_train_idx_file=self.cfg.data.train_idx[1],
                                                      a_val_idx_file=self.cfg.data.val_idx[1])
            train_sampler_2a3 = SubsetRandomSampler(indices=train_idx_2a3)
            val_sampler_2a3 = SubsetRandomSampler(indices=val_idx_2a3)
            train_loader_2a3 = DataLoader(dataset=dataset, batch_size=self.cfg.train.batch, sampler=train_sampler_2a3)
            val_loader_2a3 = DataLoader(dataset=dataset, batch_size=self.cfg.val.batch, sampler=val_sampler_2a3)
            loader_2a3 = Loader(name='2A3', train=train_loader_2a3, val=val_loader_2a3)
            self.loaders.append(loader_2a3)

    def train(self):
        self.__init_train_loader()

        for run in range(0, self.cfg.train.run):
            self.logger.info(f'Experiment Run {run} is started.')
            self.exps.append(a_cfg_name=self.cfg.cfg.name, a_run=run)
            for ep in range(0, self.cfg.train.epoch):
                self.logger.info(f"Run {run}'s Epoch {ep} is started.")
                for name, loader in self.loaders.items:
                    train_loss = self.models[name].train(a_data_loader=loader.train)
                    val_loss = self.models[name].validate(a_data_loader=loader.val)
                    self.logger.info(f"Run {run}'s Epoch {ep}'s Training Loss is {train_loss} and Validation Loss is {val_loss}.")
                    self.exps.experiment.save_epoch(a_loss=val_loss, a_epoch=ep,
                                                    a_state={'epoch': ep,
                                                             'arch': self.models[name].arch_param.name,
                                                             'state_dict': self.models[name].arch.state_dict(),
                                                             'val_loss': val_loss,
                                                             'train_loss': train_loss,
                                                             'optimizer': self.models[name].optim.state_dict()})

    def test(self):
        NotImplemented

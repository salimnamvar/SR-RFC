""" Experiment-1


"""

# region Imported Dependencies
import logging
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from brain.nn.net import Net
from brain.nn.util.param import Param
from brain.task.base import BaseTask
from brain.util.cfg.config import BrainConfig
from brain.util.data.dataset import TrainDataset, TestDataset
from brain.util.data.load import load_indices, Loaders, Loader
from brain.util.exp.state import Experiments


# endregion Imported Dependencies


class Task(BaseTask):
    def __init__(self, a_name: str = 'RFC.TASK'):
        super().__init__()
        self.name: str = a_name
        self.iteration: int = -1
        self.cfg: BrainConfig = BrainConfig.get_instance()
        self.model: Net = None
        self.loaders: Loaders = Loaders()
        self.exps: Experiments = Experiments(a_root_path=self.cfg.tsk.exp_dir)
        self.logger = logging.getLogger(self.cfg.log.name + '.' + self.name)

    def __init_model(self, a_name: str):
        # Shared Parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arch = Param(name=self.cfg.nn.arch.name, kwargs=self.cfg.nn.arch.kwargs)
        optim = Param(name=self.cfg.optim.name, kwargs=self.cfg.optim.kwargs)
        lrs = Param(name=self.cfg.lrs.name, kwargs=self.cfg.lrs.kwargs)
        loss = Param(name=self.cfg.loss.name, kwargs=self.cfg.loss.kwargs)

        self.model = Net(a_name=a_name, a_arch=arch, a_optim=optim, a_lrs=lrs, a_loss=loss, a_device=device,
                         a_half=self.cfg.nn.half)

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
            val_loader = DataLoader(dataset=dataset, batch_size=self.cfg.val.batch, sampler=val_sampler,
                                    pin_memory=True)
            loader = Loader(name='SRRFC', train=train_loader, val=val_loader)
            self.loaders.append(loader)

        else:
            # DMS Data Loaders
            train_idx_dms, val_idx_dms = load_indices(a_train_idx_file=self.cfg.data.train_idx[0],
                                                      a_val_idx_file=self.cfg.data.val_idx[0])
            train_sampler_dms = SubsetRandomSampler(indices=train_idx_dms)
            val_sampler_dms = SubsetRandomSampler(indices=val_idx_dms)
            train_loader_dms = DataLoader(dataset=dataset, batch_size=self.cfg.train.batch, sampler=train_sampler_dms,
                                          pin_memory=True)
            val_loader_dms = DataLoader(dataset=dataset, batch_size=self.cfg.val.batch, sampler=val_sampler_dms,
                                        pin_memory=True)
            loader_dms = Loader(name='DMS', train=train_loader_dms, val=val_loader_dms)
            self.loaders.append(loader_dms)

            # 2A3 Data Loaders
            train_idx_2a3, val_idx_2a3 = load_indices(a_train_idx_file=self.cfg.data.train_idx[1],
                                                      a_val_idx_file=self.cfg.data.val_idx[1])
            train_sampler_2a3 = SubsetRandomSampler(indices=train_idx_2a3)
            val_sampler_2a3 = SubsetRandomSampler(indices=val_idx_2a3)
            train_loader_2a3 = DataLoader(dataset=dataset, batch_size=self.cfg.train.batch, sampler=train_sampler_2a3,
                                          pin_memory=True)
            val_loader_2a3 = DataLoader(dataset=dataset, batch_size=self.cfg.val.batch, sampler=val_sampler_2a3,
                                        pin_memory=True)
            loader_2a3 = Loader(name='2A3', train=train_loader_2a3, val=val_loader_2a3)
            self.loaders.append(loader_2a3)

    def train(self):
        self.__init_train_loader()
        for name, loader in self.loaders.items:
            self.__init_model(a_name=name)
            for run in range(0, self.cfg.train.run):
                self.logger.info(f'Experiment Run {run} is started.')
                self.exps.append(a_cfg_name=self.cfg.cfg.name, a_run=run, a_model_name=self.model.name)
                writer = SummaryWriter(self.exps[-1].path)
                for ep in range(0, self.cfg.train.epoch):
                    self.logger.info(f"Run {run}'s Epoch {ep} is started.")
                    train_loss = self.model.train(a_data_loader=loader.train, a_writer=writer, a_epoch=ep)
                    val_loss = self.model.validate(a_data_loader=loader.val, a_writer=writer, a_epoch=ep)
                    self.logger.info(
                        f"Run {run}'s Epoch {ep}'s Training Loss is {train_loss} and Validation Loss is {val_loss}.")
                    self.exps.experiment.save_epoch(a_loss=val_loss, a_epoch=ep,
                                                    a_state={'epoch': ep,
                                                             'arch': self.model.arch_param.name,
                                                             'state_dict': self.model.arch.state_dict(),
                                                             'val_loss': val_loss,
                                                             'train_loss': train_loss,
                                                             'optimizer': self.model.optim.state_dict()})
                    writer.add_scalar('data/train_epoch_loss', train_loss, ep)
                    writer.add_scalar('data/val_epoch_loss', val_loss, ep)
                writer.close()

    def __init_test_loader(self):
        # DMS Dataset
        dataset_dms = TestDataset(a_file=self.cfg.data.test, a_exp='DMS_MaP', a_max_length=self.cfg.data.max_length,
                                  a_inc_exp_type=self.cfg.data.inc_exp_type)
        data_loader_dms = DataLoader(dataset=dataset_dms, batch_size=self.cfg.test.batch, pin_memory=True)
        loader_dms = Loader(name='DMS', test=data_loader_dms)
        self.loaders.append(loader_dms)

        # 2A3_MaP Dataset
        dataset_2a3 = TestDataset(a_file=self.cfg.data.test, a_exp='2A3_MaP', a_max_length=self.cfg.data.max_length,
                                  a_inc_exp_type=self.cfg.data.inc_exp_type)
        data_loader_2a3 = DataLoader(dataset=dataset_2a3, batch_size=self.cfg.test.batch, pin_memory=True)
        loader_2a3 = Loader(name='2A3', test=data_loader_2a3)
        self.loaders.append(loader_2a3)

    def test(self):
        self.__init_test_loader()

        with torch.no_grad():
            for name, loader in self.loaders.items:
                self.__init_model(a_name=name)
                if self.cfg.data.inc_exp_type:
                    results = self.model.test(a_weights=self.cfg.test.model[name], a_data_loader=loader.test)
                else:
                    results = self.model.test(a_weights=self.cfg.test.model[name], a_data_loader=loader.test)

                results.to_csv(self.cfg.test.output[name], header=results.keys())

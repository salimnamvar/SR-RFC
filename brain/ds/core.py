""" Dataset Dispatcher

"""

# region Imported Dependencies
import importlib
from typing import Iterable
import torch
from torch.utils.data import Dataset

from brain.ds.hdl.base import BaseDataset
from brain.util.base.arg import Param


# endregion Imported Dependencies

class DatasetDispatcher(Dataset):
    def __init__(self, a_name: str, a_ds: Param) -> None:
        self.name: str = a_name
        self.ds_param: Param = a_ds
        self.ds: BaseDataset = None
        self.__init_ds()

    def __init_ds(self):
        # Build Dataset
        try:
            module_name = f"brain.ds.hdl.{self.ds_param.name}"
            module = importlib.import_module(module_name)
            arch_class = getattr(module, 'Dataset')
        except (AttributeError, ModuleNotFoundError):
            raise ValueError('Invalid `dataset.name` is entered.')
        self.ds: BaseDataset = arch_class(**self.ds_param.kwargs)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, a_index) -> Iterable[torch.Tensor]:
        return self.ds[a_index]

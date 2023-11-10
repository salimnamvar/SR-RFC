""" Conv Dataset Handler

"""

# region Imported Dependencies
from typing import List, Tuple, Iterable

import torch

from brain.ds.hdl.base import BaseDataset
# endregion Imported Dependencies


class Dataset(BaseDataset):
    def __init__(self, a_file: str, a_max_length: int = 457, a_name: str = 'Conv_Dataset'):
        super().__init__(a_file=a_file, a_max_length=a_max_length, a_name=a_name)

    def __preprocess(self, a_sequence: str, a_reactivity: List[float], a_experiment: str) -> Iterable[torch.Tensor]:
        NotImplementedError

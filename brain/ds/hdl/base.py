""" Base Dataset Handler

"""

# region Imported Dependencies
from abc import abstractmethod
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from brain.ds.util.scheme import DatasetScheme


# endregion Imported Dependencies


class BaseDataset(Dataset):
    def __init__(self, a_file: str, a_max_length: int = 457, a_name: str = 'BaseDataset') -> None:
        self.file: str = a_file
        self.max_length: int = a_max_length
        self.name: str = a_name
        self.table = pq.read_table(self.file)
        self.dataset_scheme: DatasetScheme = DatasetScheme(a_feat=self.table.column_names)

    def __len__(self) -> int:
        return self.table.num_rows

    @abstractmethod
    def _preprocess(self, a_sequence: str, a_reactivity: List[float], a_experiment: str) -> Tuple[torch.Tensor]:
        NotImplementedError("Subclasses should implement this method.")

    def _get_sample(self, a_index: int) -> Tuple[str, List[float], str]:
        try:
            t_row = self.table.slice(a_index, 1)
            row = t_row.to_pylist()[0]
            sequence = row[self.dataset_scheme.sequence.name]
            experiment = row[self.dataset_scheme.experiment.name]
            reactivity = [row[label.name] for label in self.dataset_scheme.reactivity]
        except Exception as e:
            msg = f"{self.name}'s `__get_sample` method got an error: `{e}`"
            raise RuntimeError(msg)
        return sequence, reactivity, experiment

    def __getitem__(self, a_index: int) -> Tuple[torch.Tensor]:
        try:
            # Get sample
            sequence, reactivity, experiment = self._get_sample(a_index)

            # Preprocess sample
            data = self._preprocess(sequence, reactivity, experiment)
        except Exception as e:
            msg = f"{self.name}'s `__getitem__` method got an error: `{e}`"
            raise RuntimeError(msg)
        return data

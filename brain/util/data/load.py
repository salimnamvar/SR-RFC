""" Data Loader Functions

"""


# region Imported Dependencies
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict
from torch.utils.data import DataLoader
# endregion Imported Dependencies


def load_indices(a_train_idx_file: Union[str, List[str]],
                 a_val_idx_file: Union[str, List[str]]) -> Tuple[List[int], List[int]]:
    # Handle multiple files
    train_idx_files = [a_train_idx_file] if isinstance(a_train_idx_file, str) else a_train_idx_file
    val_idx_files = [a_val_idx_file] if isinstance(a_val_idx_file, str) else a_val_idx_file

    # Initialize lists to store the indices
    train_idx = []
    val_idx = []

    # Read and convert train indices
    for filename in train_idx_files:
        with open(filename, "r") as train_file:
            for line in train_file:
                train_idx.append(int(line.strip()))  # Convert the line to an integer and add it to the list

    # Read and convert valid indices
    for filename in val_idx_files:
        with open(filename, "r") as valid_file:
            for line in valid_file:
                val_idx.append(int(line.strip()))  # Convert the line to an integer and add it to the list

    return train_idx, val_idx


@dataclass
class Loader:
    name: str
    train: DataLoader = None
    val: DataLoader = None
    test: DataLoader = None


class Loaders:
    def __init__(self) -> None:
        self._items: Dict[str] = {}

    def append(self, a_loader: Loader) -> None:
        self._items[a_loader.name] = a_loader

    def pop(self, a_key: str) -> None:
        self._items.pop(a_key)

    @property
    def items(self) -> dict:
        return self._items

    def __getitem__(self, a_key: str) -> Loader:
        return self._items[a_key]

    def __len__(self) -> int:
        return len(self._items)

""" Data Loader Functions

"""


# region Imported Dependencies
from typing import List, Tuple, Union
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

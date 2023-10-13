""" Data Loader Functions

"""

# region Imported Dependencies
from typing import List, Tuple
# endregion Imported Dependencies


def load_indices(a_train_idx_file: str, a_val_idx_file: str) -> Tuple[List[int], List[int]]:
    # Initialize lists to store the indices
    train_idx = []
    val_idx = []

    # Read and convert train indices
    with open(a_train_idx_file, "r") as train_file:
        for line in train_file:
            train_idx.append(int(line.strip()))  # Convert the line to an integer and add it to the list

    # Read and convert valid indices
    with open(a_val_idx_file, "r") as valid_file:
        for line in valid_file:
            val_idx.append(int(line.strip()))  # Convert the line to an integer and add it to the list

    return train_idx, val_idx

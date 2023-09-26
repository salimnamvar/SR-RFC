""" Dataset

"""


# region Imported Dependencies
from torch.utils.data import Dataset
import pandas as pd
# endregion Imported Dependencies


class SRRFCDataset(Dataset):
    """ Stanford Ribonanza RNA Folding Dataset

    """

    def __init__(self, a_csv_filename: str, a_batch_size: int):
        self.csv_filename: str = a_csv_filename
        self.batch_size: int = a_batch_size
        self.reader = pd.read_csv(a_csv_filename, chunksize=a_batch_size)

    def __len__(self):
        # Get number of samples
        num_samples = sum(len(batch) for batch in self.reader)
        # Reset the reader
        self.reader = pd.read_csv(self.csv_filename, chunksize=self.batch_size)
        return num_samples

    def __getitem__(self, item):
        NotImplementedError
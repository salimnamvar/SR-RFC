""" Dataset

"""


# region Imported Dependencies
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
# endregion Imported Dependencies


class SRRFCDataset(Dataset):
    """ Stanford Ribonanza RNA Folding Dataset

    """

    def __init__(self, a_csv_filename: str, a_batch_size: int):
        self.csv_filename: str = a_csv_filename
        self.batch_size: int = a_batch_size
        self.reader = pd.read_csv(a_csv_filename, chunksize=a_batch_size)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        # Get number of samples
        num_samples = sum(len(batch) for batch in self.reader)
        # Reset the reader
        self.reader = pd.read_csv(self.csv_filename, chunksize=self.batch_size)
        return num_samples

    def __getitem__(self, item):
        # Read the next batch
        batch = next(self.reader)
        return batch


if __name__ == '__main__':
    dataset = SRRFCDataset(a_csv_filename='G:/Challenges/RNA/data/train_data.csv',
                           a_batch_size=10)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    for data in dataloader:
        x = data

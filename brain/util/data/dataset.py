""" Dataset

"""


# region Imported Dependencies
from typing import List
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
from torchvision import transforms
# endregion Imported Dependencies


class SRRFCDataset(Dataset):
    """ Stanford Ribonanza RNA Folding Dataset

    """

    def __init__(self, a_file: str, a_chunk_size: int, a_seq_cats: List[str] = ['A', 'C', 'G', 'U']):
        self.file: str = a_file
        self.chunk_size: int = a_chunk_size
        self.seq_cats: List[str] = a_seq_cats
        self.seq_cat_to_index: dict = {category: index for index, category in enumerate(self.seq_cats)}
        self.table = pq.read_table(self.file)
        self.chunks = self.table.to_batches(self.chunk_size)
        self.current_chunk_idx = 0
        self.current_chunk = self.chunks[self.current_chunk_idx].to_pandas()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.table.num_rows

    def __process_sequence(self, a_seq: str) -> np.ndarray:
        # Convert the sequence to an array of category indices
        seq_indices = np.array([self.seq_cat_to_index[cat] for cat in a_seq], dtype=int)

        # Create a one-hot encoding array
        one_hot_seq = (seq_indices[:, None] == np.arange(len(self.seq_cats))).astype(int)
        return one_hot_seq

    def __getitem__(self, a_index):
        try:
            # Data sample's inner-chunk index
            in_chunk_index = a_index % len(self.current_chunk)

            # Iterate Chunk
            if a_index % len(self.current_chunk) == 0:
                self.current_chunk_idx += 1
                self.current_chunk = self.chunks[self.current_chunk_idx].to_pandas()

            sequence = self.__process_sequence(self.current_chunk.iloc[in_chunk_index]['sequence'])
        except Exception as e:
            raise e
        return sequence


if __name__ == '__main__':
    dataset = SRRFCDataset(a_file='G:/Challenges/RNA/data/train_data.parquet', a_chunk_size=1000)
    dataloader = DataLoader(dataset=dataset, batch_size=100)
    for i, data in enumerate(dataloader):
        print(f'Data Sample Index: {i}')

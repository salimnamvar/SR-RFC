""" SRRFC Dataset Handler

"""


# region Imported Dependencies
import time
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
from torchvision import transforms
from brain.util.data.scheme import DatasetScheme, SampleIndex, SampleIndices
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
        self.sample_indices = SampleIndices()
        self.__init_samples()
        self.dataset_scheme: DatasetScheme = DatasetScheme(a_feat=self.chunks[0].to_pandas().columns.tolist())
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __init_samples(self):
        sample_counter = 0
        for chunk_id, chunk in enumerate(self.chunks):
            for in_chunk_id in range(len(chunk)):
                id = sample_counter + in_chunk_id
                self.sample_indices.append(SampleIndex(idx=id, chunk_idx=chunk_id, in_chunk_idx=in_chunk_id))
            sample_counter += len(chunk)

    def __len__(self):
        return self.table.num_rows

    def __preprocess(self, a_sequence: str, a_reactivity: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        sequence = self.__proc_sequence(a_sequence)
        reactivity = self.__proc_reactivity(a_reactivity, a_seq_len=len(a_sequence))
        return sequence, reactivity

    def __proc_sequence(self, a_seq: str) -> np.ndarray:
        # Convert the sequence to an array of category indices
        seq_indices = np.array([self.seq_cat_to_index[cat] for cat in a_seq], dtype=int)

        # Create a one-hot encoding array
        one_hot_seq = (seq_indices[:, None] == np.arange(len(self.seq_cats))).astype(int)
        return one_hot_seq

    def __proc_reactivity(self, a_react: List[float], a_seq_len: int) -> np.ndarray:
        return np.array(a_react[:a_seq_len], dtype=float)

    def __get_sample(self, a_index: int) -> Tuple[str, List[float]]:
        smp = self.sample_indices[a_index]
        sequence = self.chunks[smp.chunk_idx][self.dataset_scheme.input][smp.in_chunk_idx].as_py()
        reactivity = [self.chunks[smp.chunk_idx][label][smp.in_chunk_idx].as_py() for label in
                      self.dataset_scheme.label]
        return sequence, reactivity

    def __getitem__(self, a_index):
        try:
            # Get sample
            sequence, reactivity = self.__get_sample(a_index)

            # Preprocess sample
            sequence, reactivity = self.__preprocess(sequence, reactivity)
        except Exception as e:
            raise e
        return sequence, reactivity


if __name__ == '__main__':
    if False:
        t1 = time.time()
        dataset = SRRFCDataset(a_file='G:/Challenges/RNA/data/train_data.parquet', a_chunk_size=1000)
        dataloader = DataLoader(dataset=dataset, batch_size=1000, shuffle=True,
                                collate_fn=lambda batch: tuple(zip(*batch)))
        for i, (x_data, y_data) in enumerate(dataloader):
            print(f'Data Sample Index: {i}')
        t2 = time.time()
        print(f'Dataloader time: {t2 - t1}')

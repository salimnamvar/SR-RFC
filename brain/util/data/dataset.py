""" SRRFC Dataset Handler

"""


# region Imported Dependencies
from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from transformers import BertTokenizer
from brain.util.data.scheme import DatasetScheme, SampleIndex, SampleIndices
# endregion Imported Dependencies


class SRRFCDataset(Dataset):
    """ Stanford Ribonanza RNA Folding Dataset

    """

    def __init__(self, a_file: str, a_chunk_size: int, a_max_length: int = 457):
        self.file: str = a_file
        self.chunk_size: int = a_chunk_size
        self.max_length: int = a_max_length
        self.table = pq.read_table(self.file)
        self.chunks = self.table.to_batches(self.chunk_size)
        self.sample_indices = SampleIndices()
        self.__init_samples()
        self.dataset_scheme: DatasetScheme = DatasetScheme(a_feat=self.chunks[0].to_pandas().columns.tolist())
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def __init_samples(self):
        sample_counter = 0
        for chunk_id, chunk in enumerate(self.chunks):
            for in_chunk_id in range(len(chunk)):
                id = sample_counter + in_chunk_id
                self.sample_indices.append(SampleIndex(idx=id, chunk_idx=chunk_id, in_chunk_idx=in_chunk_id))
            sample_counter += len(chunk)

    def __len__(self):
        return self.table.num_rows

    def __preprocess(self, a_sequence: str, a_reactivity: List[float]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        reactivity = torch.tensor(np.array(a_reactivity[:len(a_sequence)], dtype=float))

        # Sequence Tokenization
        sequence = ' '.join(a_sequence)
        inputs = self.tokenizer.encode_plus(text=sequence, text_pair=None, add_special_tokens=True,
                                            max_length=self.max_length,
                                            padding='max_length', return_token_type_ids=True,
                                            truncation=True)
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)

        # Reactivity Nan values
        nan_ids = torch.where(torch.isnan(reactivity))
        # Plus one index more for passing the classification token which is in the index 0
        attention_mask[nan_ids[0] + 1] = 0
        reactivity[nan_ids] = 0

        # Reactivity Padding
        padded_reactivity = torch.zeros(self.max_length)
        padded_reactivity[:len(reactivity)] = reactivity

        return input_ids, attention_mask, token_type_ids, padded_reactivity

    def __get_sample(self, a_index: int) -> Tuple[str, List[float]]:
        smp = self.sample_indices[a_index]
        sequence = self.chunks[smp.chunk_idx][self.dataset_scheme.input][smp.in_chunk_idx].as_py()
        reactivity = [self.chunks[smp.chunk_idx][label][smp.in_chunk_idx].as_py() for label in
                      self.dataset_scheme.label]
        return sequence, reactivity

    def __getitem__(self, a_index) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        try:
            # Get sample
            sequence, reactivity = self.__get_sample(a_index)

            # Preprocess sample
            input_ids, attention_mask, token_type_ids, reactivity = self.__preprocess(sequence, reactivity)
        except Exception as e:
            raise e
        return input_ids, attention_mask, token_type_ids, reactivity


if __name__ == '__main__':
    if False:
        dataset = SRRFCDataset(a_file='G:/Challenges/RNA/data/train_data.parquet', a_chunk_size=1000)
        dataloader = DataLoader(dataset=dataset, batch_size=1000, shuffle=True,
                                collate_fn=lambda batch: tuple(zip(*batch)))
        for i, (input_ids, attention_mask, token_type_ids, reactivity) in enumerate(dataloader):
            print(f'Data Sample Index: {i}')

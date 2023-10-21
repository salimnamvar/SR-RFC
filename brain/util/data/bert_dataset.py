""" SRRFC Parquet-based BERT Dataset Handler

"""


# region Imported Dependencies
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from transformers import BertTokenizer
from brain.util.data.scheme import DatasetScheme
# endregion Imported Dependencies


class TrainDataset(Dataset):
    """ Stanford Ribonanza RNA Folding Training Dataset

    """

    def __init__(self, a_file: str, a_max_length: int = 457, a_inc_exp_type: bool = False):
        self.file: str = a_file
        self.max_length: int = a_max_length
        self.inc_exp_type: bool = a_inc_exp_type
        self.table = pq.read_table(self.file)
        self.dataset_scheme: DatasetScheme = DatasetScheme(a_feat=self.table.column_names)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.exp_map: dict = {'DMS_MaP': '0',
                              '2A3_MaP': '1'}
        sequence_mapper = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
        self.sequence_mapper = defaultdict(lambda: 0, sequence_mapper)

    def __len__(self):
        return self.table.num_rows

    def __preprocess(self, a_sequence: str, a_reactivities: List[float]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        sequence_length = len(a_sequence)
        input_ids = np.zeros(self.max_length, dtype=np.int64)
        attention_mask = np.zeros(self.max_length, dtype=np.int64)
        token_ids = np.zeros(self.max_length, dtype=np.int64)
        reactivities = np.zeros(self.max_length, dtype=np.float64)

        input_ids[: sequence_length] = [self.sequence_mapper[letter] for letter in a_sequence]
        attention_mask[: sequence_length] = [1 if value is not None else 0 for value in a_reactivities[:sequence_length]]
        reactivities[: sequence_length] = a_reactivities[:sequence_length]

        input_ids = torch.from_numpy(input_ids)
        attention_mask = torch.from_numpy(attention_mask)
        token_ids = torch.from_numpy(token_ids)
        reactivities = torch.from_numpy(reactivities)

        nan_ids = torch.where(torch.isnan(reactivities))
        reactivities[nan_ids] = 0
        return input_ids, attention_mask, token_ids, reactivities

    def __get_sample(self, a_index: int) -> Tuple[str, List[float]]:
        row = self.table.slice(a_index, 1).to_pylist()[0]
        sequence = row[self.dataset_scheme.input.name]
        reactivity = [row[label.name] for label in self.dataset_scheme.label]
        return sequence, reactivity

    def __getitem__(self, a_index) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        try:
            # Get sample
            sequence, reactivity = self.__get_sample(a_index)

            # Preprocess sample
            input_ids, attention_mask, token_type_ids, reactivity = self.__preprocess(sequence, reactivity)
        except Exception as e:
            raise e
        return (input_ids, attention_mask, token_type_ids), reactivity

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
from brain.ds.util.scheme import DatasetScheme
# endregion Imported Dependencies


class TrainDataset(Dataset):
    """ Stanford Ribonanza RNA Folding Training Dataset

    """

    def __init__(self, a_file: str, a_max_length: int = 457, a_inc_exp_type: bool = False, a_one_hot: bool = False):
        self.file: str = a_file
        self.one_hot: bool = a_one_hot
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
        input_ids = torch.zeros(self.max_length, dtype=torch.int64)
        attention_mask = torch.ones(self.max_length, dtype=torch.int64)
        token_ids = torch.zeros(self.max_length, dtype=torch.int64)
        reactivities = torch.zeros(self.max_length, dtype=torch.float64)

        reactivity = torch.tensor(np.array(a_reactivities[:len(a_sequence)], dtype=float))
        seq = torch.tensor([self.sequence_mapper[letter] for letter in a_sequence])

        nan_ids = torch.isnan(reactivity)
        seq = seq[~nan_ids]
        react = reactivity[~nan_ids]
        sequence_length = len(seq)

        input_ids[: sequence_length] = seq
        reactivities[: sequence_length] = react
        return input_ids, attention_mask, token_ids, reactivities

    def __postprocess(self, a_input_ids) -> Tensor:
        if self.one_hot:
            a_input_ids = torch.eye(len(self.sequence_mapper) + 1)[a_input_ids]
        return a_input_ids

    def __get_sample(self, a_index: int) -> Tuple[str, List[float]]:
        row = self.table.slice(a_index, 1).to_pylist()[0]
        sequence = row[self.dataset_scheme.sequence.name]
        reactivity = [row[label.name] for label in self.dataset_scheme.reactivity]
        return sequence, reactivity

    def __getitem__(self, a_index) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        try:
            # Get sample
            sequence, reactivity = self.__get_sample(a_index)

            # Preprocess sample
            input_ids, attention_mask, token_type_ids, reactivity = self.__preprocess(sequence, reactivity)

            # Postprocess sample
            input_ids = self.__postprocess(input_ids)
        except Exception as e:
            raise e
        return (input_ids, attention_mask, token_type_ids), reactivity

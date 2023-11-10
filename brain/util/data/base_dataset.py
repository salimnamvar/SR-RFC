""" SRRFC Parquet-based Dataset Handler

"""

# region Imported Dependencies
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

    def __init__(self, a_file: str, a_max_length: int = 457, a_inc_exp_type: bool = False):
        self.file: str = a_file
        self.max_length: int = a_max_length
        self.inc_exp_type: bool = a_inc_exp_type
        self.table = pq.read_table(self.file)
        self.dataset_scheme: DatasetScheme = DatasetScheme(a_feat=self.table.column_names)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,)
        self.exp_map: dict = {'DMS_MaP': '0',
                              '2A3_MaP': '1'}

    def __len__(self):
        return self.table.num_rows

    def __preprocess(self, a_sequence: str, a_reactivity: List[float],
                     a_experiment: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Sequence Tokenization
        sequence_1 = ' '.join(a_sequence)
        sequence_2 = self.exp_map[a_experiment] if self.inc_exp_type else None
        inputs = self.tokenizer.encode_plus(text=sequence_1, text_pair=sequence_2, add_special_tokens=True,
                                            max_length=self.max_length, padding='max_length',
                                            return_token_type_ids=True, truncation=True)
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int)

        # Convert the reactivity into tensor
        reactivity = torch.tensor(np.array(a_reactivity[:len(a_sequence)], dtype=np.float32))

        # Check for NaN values and create a mask to identify them
        nan_mask = torch.isnan(reactivity)

        # Calculate the minimum and maximum values for non-NaN elements
        non_nan_values = reactivity[~nan_mask]

        # Normalize the non-NaN values between 0 and 1
        # min_value = non_nan_values.min()
        # max_value = non_nan_values.max()
        # (non_nan_values - min_value) / (max_value - min_value)
        normalized = torch.zeros(reactivity.size(), dtype=torch.float)
        normalized[~nan_mask] = torch.clamp(non_nan_values, min=0, max=1)

        # NaN Values
        normalized[nan_mask] = 0

        # Reactivity Padding
        # torch.full((self.max_length,), -2, dtype=torch.float32)
        padded_reactivity = torch.zeros(self.max_length, dtype=torch.float)
        padded_reactivity[:len(normalized)] = normalized

        return input_ids, attention_mask, token_type_ids, padded_reactivity

    def __get_sample(self, a_index: int) -> Tuple[str, List[float], str]:
        t_row = self.table.slice(a_index, 1)
        row = t_row.to_pylist()[0]
        sequence = row[self.dataset_scheme.sequence.name]
        experiment = row[self.dataset_scheme.experiment.name]
        reactivity = [row[label.name] for label in self.dataset_scheme.reactivity]
        return sequence, reactivity, experiment

    def __getitem__(self, a_index) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]:
        try:
            # Get sample
            sequence, reactivity, experiment = self.__get_sample(a_index)

            # Preprocess sample
            input_ids, attention_mask, token_type_ids, reactivity = self.__preprocess(sequence, reactivity, experiment)
        except Exception as e:
            raise e
        return (input_ids, attention_mask, token_type_ids), reactivity


class TestDataset(Dataset):
    def __init__(self, a_file: str, a_exp: str, a_max_length: int = 457, a_inc_exp_type: bool = False):
        self.file: str = a_file
        self.exp: str = a_exp
        self.max_length: int = a_max_length
        self.inc_exp_type: bool = a_inc_exp_type
        self.table = pq.read_table(self.file)
        self.exp_map: dict = {'DMS_MaP': '0',
                              '2A3_MaP': '1'}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def __len__(self):
        return self.table.num_rows

    def __preprocess(self, a_sequence: str) -> Tuple[Tensor, Tensor, Tensor]:
        # Sequence Tokenization
        sequence_1 = ' '.join(a_sequence)
        sequence_2 = self.exp_map[self.exp] if self.inc_exp_type else None
        inputs = self.tokenizer.encode_plus(text=sequence_1, text_pair=sequence_2, add_special_tokens=True,
                                            max_length=self.max_length, padding='max_length',
                                            return_token_type_ids=True, truncation=True)
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int)
        return input_ids, attention_mask, token_type_ids

    def __get_sample(self, a_index: int) -> Tuple[int, int, str]:
        t_row = self.table.slice(a_index, 1)
        row = t_row.to_pylist()[0]
        id_min = row['id_min']
        id_max = row['id_max']
        sequence = row['sequence']
        return id_min, id_max, sequence

    def __getitem__(self, a_index) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[int, int]]:
        try:
            # Get sample
            id_min, id_max, sequence = self.__get_sample(a_index)

            # Preprocess sample
            input_ids, attention_mask, token_type_ids = self.__preprocess(sequence)
        except Exception as e:
            raise e
        return (input_ids, attention_mask, token_type_ids), (id_min, id_max)

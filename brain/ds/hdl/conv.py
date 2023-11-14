""" Conv Dataset Handler

"""

# region Imported Dependencies
from typing import List, Tuple
import torch
import torch.nn.functional as F
from brain.ds.hdl.base import BaseDataset
# endregion Imported Dependencies


class Dataset(BaseDataset):
    def __init__(self, a_file: str, a_max_length: int = 484, a_name: str = 'Conv_Dataset'):
        super().__init__(a_file=a_file, a_max_length=a_max_length, a_name=a_name)
        self.char_map = {'A': 0.2, 'C': 0.4, 'G': 0.6, 'U': 0.8}

    def _process_rna(self, a_sequence: str) -> torch.Tensor:
        # Convert characters to numeric values
        numeric_sequence = torch.tensor([self.char_map[char] for char in a_sequence], dtype=torch.float32)

        # Pad the sequence to a length of 484
        padded_sequence = F.pad(numeric_sequence, (0, self.max_length - len(numeric_sequence)))

        # Reshape the padded sequence into a 22x22 matrix
        rna = padded_sequence.view(1, 22, 22)

        return rna

    def _process_reactivity(self, a_reactivity: List[float]) -> torch.Tensor:
        # Handle None values by replacing them with zeros
        processed_reactivity = [0.0 if value is None else value for value in a_reactivity]

        # Normalize the list between 0 and 1
        normalized_reactivity = torch.tensor(processed_reactivity, dtype=torch.float32) / torch.max(
            torch.tensor(processed_reactivity, dtype=torch.float32))

        # Pad the list to the target length
        padded_reactivity = F.pad(normalized_reactivity, (0, self.max_length - len(normalized_reactivity)))

        # Reshape the padded reactivity into a 22x22 matrix
        reactivity = padded_reactivity.view(1, 22, 22)

        return reactivity

    def _preprocess(self, a_sequence: str, a_reactivity: List[float], a_experiment: str) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # Process RNA sequence
        rna = self._process_rna(a_sequence=a_sequence)

        # Process Reactivity values
        reactivity = self._process_reactivity(a_reactivity=a_reactivity)

        return rna, reactivity

    def __getitem__(self, a_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Get sample
            sequence, reactivity, experiment = self._get_sample(a_index)

            # Preprocess sample
            processed_rna, processed_reactivity = self._preprocess(sequence, reactivity, experiment)
        except Exception as e:
            msg = f"{self.name}'s `__getitem__` method got an error: `{e}`"
            raise RuntimeError(msg)
        return processed_rna, processed_reactivity

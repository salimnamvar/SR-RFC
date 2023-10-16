""" Dataset Scheme

"""


# region Imported Dependencies
from dataclasses import dataclass
from typing import List
# endregion Imported Dependencies


@dataclass
class DatasetScheme:
    features: List[str]
    label: List[int]
    input: int

    def __init__(self, a_feat: List[str]) -> None:
        self.features = a_feat
        self.label = [i for i, col in enumerate(self.features) if 'reactivity' in col and 'error' not in col]
        self.input = self.features.index('sequence')
        self.experiment = self.features.index('experiment_type')


@dataclass
class SampleIndex:
    idx: int
    chunk_idx: int
    in_chunk_idx: int


class SampleIndices:
    def __init__(self) -> None:
        self._items = []

    def append(self, a_item: SampleIndex) -> None:
        self._items.append(a_item)

    @property
    def items(self) -> List[SampleIndex]:
        return self._items

    def __getitem__(self, a_index: int) -> SampleIndex:
        return self._items[a_index]

    def __len__(self) -> int:
        return len(self._items)
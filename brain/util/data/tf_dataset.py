""" SRRFC TFRecord-based Dataset Handler

"""
import os
import time
# region Imported Dependencies
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
import tensorflow as tf


# endregion Imported Dependencies

class TrainDataset(Dataset):
    def __init__(self, a_file: str, a_dataset_length: int = 1_643_680):
        self.file: str = a_file
        self.length: int = a_dataset_length
        self.data: tf.data.TFRecordDataset = tf.data.TFRecordDataset(self.file)

    def __len__(self):
        return self.length

    def __get_sample(self, a_index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        t1 = time.time()
        # Read example
        for raw_record in self.data.take(a_index + 1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

        # Extract features
        input_ids = torch.tensor(example.features.feature['input_ids'].int64_list.value)
        attention_mask = torch.tensor(example.features.feature['attention_mask'].int64_list.value)
        token_ids = torch.tensor(example.features.feature['token_ids'].int64_list.value)
        reactivities = torch.tensor(example.features.feature['reactivities'].float_list.value)
        print(f"Reading time is {time.time() - t1}")
        return input_ids, attention_mask, token_ids, reactivities

    def __getitem__(self, a_index):
        input_ids, attention_mask, token_ids, reactivities = self.__get_sample(a_index=a_index)
        return input_ids, attention_mask, token_ids, reactivities


if __name__ == '__main__':
    ds = TrainDataset(a_file="G:\Challenges\RNA\data\\train_data.tfrecord")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, pin_memory=True, num_workers=0)
    for input_ids, attention_mask, token_ids, reactivities in dataloader:
        print(len(input_ids))

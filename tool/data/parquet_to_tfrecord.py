""" Convert Parquet File to TFRecord

"""

# region IMPORT
import argparse
from collections import defaultdict

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import tensorflow as tf
from tqdm import tqdm


# endregion IMPORT


# region FUNCTION
def load_data(a_parquet_file: str) -> pyarrow.Table:
    # Select Data
    table = pq.read_table(a_parquet_file)
    target_columns = [col for i, col in enumerate(table.column_names) if 'reactivity' in col and 'error' not in col]
    input_columns = ['sequence', 'experiment_type']
    columns = input_columns + target_columns
    table = table.select(columns)
    return table


def tokenize(a_sequence: str, a_reactivities: list, a_max_length=457):
    sequence_length = len(a_sequence)
    sequence_mapper = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
    sequence_mapper = defaultdict(lambda: 0, sequence_mapper)

    input_ids = np.zeros(a_max_length, dtype=np.int64)
    attention_mask = np.zeros(a_max_length, dtype=np.int64)
    token_ids = np.zeros(a_max_length, dtype=np.int64)
    reactivities = np.zeros(a_max_length, dtype=np.float64)

    input_ids[: sequence_length] = [sequence_mapper[letter] for letter in a_sequence]
    attention_mask[: sequence_length] = [1 if value is not None else 0 for value in a_reactivities[:sequence_length]]
    reactivities[: sequence_length] = a_reactivities[:sequence_length]
    return input_ids, attention_mask, token_ids, reactivities


def convert_to_tfexample(a_row: dict, a_max_length: int = 457) -> tf.train.Example:
    ractivities = [value for key, value in a_row.items() if 'reactivity' in key]
    input_ids, attention_mask, token_ids, reactivities = tokenize(a_sequence=a_row['sequence'],
                                                                  a_reactivities=ractivities,
                                                                  a_max_length=a_max_length)
    features = {
        'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
        'attention_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=attention_mask)),
        'token_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=token_ids)),
        'reactivities': tf.train.Feature(float_list=tf.train.FloatList(value=reactivities))
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def parquet2tfrecord(a_parquet_file: str, a_tfrecord_file: str):
    table: pyarrow.Table = load_data(a_parquet_file=a_parquet_file)
    with tf.io.TFRecordWriter(a_tfrecord_file) as writer:
        for i in tqdm(range(table.num_rows), desc='Converting samples into TFRecord '):
            row = table.slice(i, 1).to_pylist()[0]
            example = convert_to_tfexample(row)
            writer.write(example.SerializeToString())
# endregion FUNCTION


# region Tool
def main():
    parser = argparse.ArgumentParser(description='Convert Parquet File to TFRecord')
    parser.add_argument('--a_parquet_file', default="G:\Challenges\RNA\data\\train_data.parquet",
                        type=str, help='Input Parquet File Path')
    parser.add_argument('--a_tfrecord_file', default="G:\Challenges\RNA\data\\train_data.tfrecord",
                        type=str, help='Output TFRecord File Path')
    args = parser.parse_args()

    parquet2tfrecord(a_parquet_file=args.a_parquet_file, a_tfrecord_file=args.a_tfrecord_file)


if __name__ == '__main__':
    main()
# endregion Tool

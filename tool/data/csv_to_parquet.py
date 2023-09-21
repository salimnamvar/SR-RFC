""" Convert CSV to Parquet

"""


# region IMPORT
import argparse
from typing import NoReturn
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# endregion IMPORT


def csv2parquet(a_csv_path: str, a_parquet_path: str, a_compression: str = 'snappy') -> NoReturn:
    df = pd.read_csv(a_csv_path)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, a_parquet_path, compression=a_compression)


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to Parquet')
    parser.add_argument('a_csv_path', type=str, help='Input CSV file path')
    parser.add_argument('a_parquet_path', type=str, help='Output Parquet file path')
    parser.add_argument('--a_compression', type=str, default='snappy',
                        help='Compression algorithm (default: snappy)')
    args = parser.parse_args()

    csv2parquet(a_csv_path=args.a_csv_path, a_parquet_path=args.a_parquet_path, a_compression=args.a_compression)


if __name__ == '__main__':
    main()

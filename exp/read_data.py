""" Experiment of Read Data

"""


# region IMPORT
import pyarrow.parquet as pq
# endregion IMPORT


parquet_file = "G:/Challenges/RNA/data/test_sequences.parquet"
table = pq.read_table(parquet_file)
df = table.to_pandas()


import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

table = pq.read_table("G:\Challenges\RNA\data\\train_data.parquet")

length = []
for i in tqdm(range(0, table.num_rows)):
    row = table.slice(i, 1)
    sequence = row['sequence'].to_pylist()[0]
    length.append(len(sequence))

print(f"Max length is {np.max(length)}")

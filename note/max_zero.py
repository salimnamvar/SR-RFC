""" Find Zero Reactivity
    Find the samples that maximum reactivity value of them is zero.
"""

# region Imported Dependencies
import csv
from tqdm import tqdm
from brain.util.data.base_dataset import TrainDataset
# endregion Imported Dependencies


ds = TrainDataset(a_file='G:/Challenges/RNA/data/train_data.parquet')
csv_file_path = "G:/Challenges/RNA/code/SR-RFC/data/zero_samples.csv"

with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for i in tqdm(range(0, ds.table.num_rows)):
        t_row = ds.table.slice(i, 1)
        row = t_row.to_pylist()[0]
        reactivity = [row[label.name] for label in ds.dataset_scheme.reactivity]
        processed = [0.0 if value is None else value for value in reactivity]
        if max(processed) == 0:
            csv_writer.writerow([i])

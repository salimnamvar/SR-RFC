from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr

from brain.util.data.base_dataset import TrainDataset
import matplotlib.pyplot as plt

ds = TrainDataset(a_file='G:/Challenges/RNA/data/train_data.parquet')
x = ds.table.slice(0, 1)
row = x.to_pylist()[0]

sequence = row[ds.dataset_scheme.sequence.name]
sequence_mapper = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
sequence_mapper = defaultdict(lambda: 0, sequence_mapper)
sequence = np.array([sequence_mapper[letter] for letter in sequence])


experiment = row[ds.dataset_scheme.experiment.name]
error = np.array([row[label.name] for label in ds.dataset_scheme.reactivity_error])

reactivity = np.array([row[label.name] for label in ds.dataset_scheme.reactivity])
reactivity[np.where(reactivity == None)] = 0

pad_seq = np.zeros_like(reactivity)
pad_seq[:len(sequence)] = sequence

correlation, _ = pearsonr(pad_seq, reactivity)
print("Pearson Correlation:", correlation)

plt.plot(pad_seq, reactivity, label='Sequence', marker='o')
#plt.scatter(pad_seq, reactivity, label='Data Points')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()

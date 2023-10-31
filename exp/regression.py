from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr

from brain.util.data.base_dataset import TrainDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

X = pad_seq.reshape(-1, 1)  # Reshape for single feature
y = reactivity

model = LinearRegression()
model.fit(X, y)

plt.scatter(pad_seq, reactivity, label='Data Points')
plt.plot(pad_seq, model.predict(X), color='red', label='Regression Line')
plt.xlabel('RNA Sequence Levels')
plt.ylabel('Reactivity Values')
plt.legend()
plt.show()
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from brain.util.data.base_dataset import TrainDataset

# Enable interactive mode for updating the same plot
plt.ion()

# Create a figure and axis
fig, ax = plt.subplots()


def plot_sequence_reactivities(sequence, reactivities, errors, label):
    ax.clear()  # Clear the existing plot

    # Filter out None values and clip reactivity values within the range [0, 1]
    filtered_reactivities = [val for val in reactivities if val is not None]
    clipped_reactivities = np.clip(filtered_reactivities, 0, 1)

    # Create a list to store clipped and unclipped reactivities
    combined_reactivities = []
    idx_filtered = 0

    # Iterate through the original reactivities
    for val in reactivities:
        if val is not None:
            # Use the clipped value for non-None values
            combined_reactivities.append(clipped_reactivities[idx_filtered])
            idx_filtered += 1
        else:
            # Use None for None values
            combined_reactivities.append(None)

    # Plot the data points with valid float values
    valid_indices = [i for i, val in enumerate(combined_reactivities) if val is not None]
    valid_reactivities = [val for val in combined_reactivities if val is not None]
    ax.scatter(valid_indices, valid_reactivities, marker='o', color='blue', label=f'Clipped Reactivities ({label})')

    # Highlight the None values with a different marker (e.g., 'x') and color
    none_indices = [i for i, val in enumerate(combined_reactivities) if val is None]
    none_reactivities = [None] * len(none_indices)
    ax.scatter(none_indices, none_reactivities, marker='x', color='red', label=f'None Values ({label})')

    # Plot error points as red points
    error_indices = [i for i, val in enumerate(errors) if val is not None]
    error_values = [val for val in errors if val is not None]
    #ax.scatter(error_indices, error_values, marker='o', color='red', label=f'Errors ({label})')

    # Set labels for the axes
    ax.set_xlabel('Sequence')
    ax.set_ylabel('Clipped Reactivities (0-1)')

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(sequence)))
    ax.set_xticklabels(list(sequence))

    # Add grid lines to help distinguish characters
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')

    # Add a legend
    ax.legend()

    # Introduce a delay to observe the changes
    plt.pause(1.0)  # Adjust the delay time as needed (1.0 seconds in this example)


def plot_sequence_reactivities_norm(sequence, reactivities, errors, label):
    ax.clear()  # Clear the existing plot

    # Normalize reactivity values between 0 and 1
    min_reactivity = min(val for val in reactivities if val is not None)
    max_reactivity = max(val for val in reactivities if val is not None)
    normalized_reactivities = [(val - min_reactivity) / (max_reactivity - min_reactivity) if val is not None else None for val in reactivities]

    # Plot the data points with valid float values
    valid_indices = [i for i, val in enumerate(normalized_reactivities) if val is not None]
    valid_reactivities = [val for val in normalized_reactivities if val is not None]
    ax.scatter(valid_indices, valid_reactivities, marker='o', color='blue', label=f'Normalized Reactivities ({label})')

    # Highlight the None values with a different marker (e.g., 'x') and color
    none_indices = [i for i, val in enumerate(normalized_reactivities) if val is None]
    none_reactivities = [None] * len(none_indices)
    ax.scatter(none_indices, none_reactivities, marker='x', color='red', label=f'None Values ({label})')

    # Plot error points as red points
    error_indices = [i for i, val in enumerate(errors) if val is not None]
    error_values = [val for val in errors if val is not None]
    #ax.scatter(error_indices, error_values, marker='o', color='red', label=f'Errors ({label})')

    # Set labels for the axes
    ax.set_xlabel('Sequence')
    ax.set_ylabel('Normalized Reactivities (0-1)')

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(sequence)))
    ax.set_xticklabels(list(sequence))

    # Add grid lines to help distinguish characters
    ax.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')

    # Add a legend
    ax.legend()

    # Introduce a delay to observe the changes
    plt.pause(1.0)  # Adjust the delay time as needed (1.0 seconds in this example)


ds = TrainDataset(a_file='G:/Challenges/RNA/data/train_data.parquet')
for i in tqdm(range(0, ds.table.num_rows)):
    x = ds.table.slice(i, 1)
    row = x.to_pylist()[0]
    sequence = row[ds.dataset_scheme.sequence.name]
    reactivities = np.array([row[label.name] for label in ds.dataset_scheme.reactivity])[:len(sequence)]
    errors = np.array([row[label.name] for label in ds.dataset_scheme.reactivity_error])[:len(sequence)]
    plot_sequence_reactivities(sequence, reactivities, errors, f'Sequence-{i}')

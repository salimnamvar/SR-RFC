""" Data Collate Functions

"""


# region Imported Dependencies
from typing import Any
from torch.nn.utils.rnn import pad_sequence, pack_sequence
# endregion Imported Dependencies


def collate_sequences(a_batch: Any):
    # 'a_batch' is a  list of (sequence, label) pairs
    sequences, labels = zip(*a_batch)

    # Pad sequences to the length of the longest sequence
    padded_sequences = pad_sequence(sequences, batch_first=True)

    # Pad labels to the length of the longest label
    padded_labels = pad_sequence(labels, batch_first=True)

    # Create a mask to handle padding
    sequence_mask = (padded_sequences != 0).float()
    label_mask = (padded_labels != 0).float()

    # Pack the sequences and labels into PackedSequence objects
    packed_sequences = pack_sequence(sequences, enforce_sorted=False)
    packed_labels = pack_sequence(labels, enforce_sorted=False)

    return padded_sequences, padded_labels, sequence_mask, label_mask, packed_sequences, packed_labels
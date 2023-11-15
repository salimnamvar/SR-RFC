""" LSTM v0 - Architectures

"""

# region Imported Dependencies
from torch import nn, Tensor
# endregion Imported Dependencies


class Arch(nn.Module):
    def __init__(self, a_max_length: int = 457, a_out_size: int = 457) -> None:
        super(Arch, self).__init__()
        self.lstm = nn.LSTM(a_max_length, 64, 2)
        self.fc = nn.Linear(64, a_out_size)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (sequence_length, batch_size, input_size)
        lstm_out, _ = self.lstm(x)  # Add an extra dimension for input_size

        # Use the output from the last time step for each sequence
        lstm_out = lstm_out[-1, :]

        # Fully connected layer
        output = self.fc(lstm_out)
        return output

""" LSTM v0 - Architectures

"""

# region Imported Dependencies
import torch
from torch import nn, Tensor
# endregion Imported Dependencies


class Arch(nn.Module):
    def __init__(self, a_max_length: int = 457, a_out_size: int = 457) -> None:
        super(Arch, self).__init__()
        self.sequence_length = a_max_length
        self.input_size = 5
        self.hidden_size = 5
        self.num_layers = 10
        self.output_size = a_out_size

        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dropout: nn.Dropout = nn.Dropout(0.3)

        # Fully connected layer to map LSTM output to the desired output shape
        self.fc = nn.Linear(self.hidden_size * self.sequence_length, self.output_size)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, a_input_ids: torch.Tensor, a_attention_mask: torch.Tensor,
                a_token_type_ids: torch.Tensor) -> Tensor:
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, a_input_ids.size(0), self.hidden_size).to(a_input_ids.device)
        c0 = torch.zeros(self.num_layers, a_input_ids.size(0), self.hidden_size).to(a_input_ids.device)

        # Forward pass through the LSTM layer
        out, _ = self.lstm(a_input_ids.to(torch.float32), (h0, c0))

        # Pass the LSTM output through the fully connected layer
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

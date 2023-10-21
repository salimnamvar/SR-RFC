""" BERTv0 - Architectures

"""

# region Imported Dependencies
from typing import Tuple
from torch import nn, Tensor
from brain.nn.bert.BB import BERT
# endregion Imported Dependencies


class Arch(nn.Module):
    """ BERTv0 Architecture - v.0

        This class is used to define the network topology that uses a BERTv0 module as a block.
    """

    def __init__(self, a_in_size: Tuple[int, int] = (457, 4), a_out_size: int = 457) -> None:
        """ Constructor

        :param a_in_size: A tuple that specifies the input shape of the network as (S, F).
            - S is an integer as sequence length.
            - F is an integer indicating the number of features.
        :param a_out_size: An integer that specifies the number of output features.
        """
        super(Arch, self).__init__()
        self.seq_len = a_in_size[0]
        self.n_input_feat = a_in_size[1]
        self.hidden_size = 4
        self.n_layers = 1
        self.n_attn_heads = 4
        self.n_output = a_out_size
        self.bert = BERT(self.n_input_feat, self.seq_len, a_hidden=self.hidden_size, a_n_layers=self.n_layers,
                         a_attn_heads=self.n_attn_heads)
        self.fc_action = nn.Linear(self.hidden_size, a_out_size)

    def forward(self, a_net_input: Tensor) -> Tensor:
        """ Feedforward

        :param a_net_input: A pytorch tensor that specifies the input of the network in shape of [B, S, F].
            - B is an integer number that indicates the Batch size as number of input samples.
            - S is an integer as sequence length.
            - F is an integer as number of features.
        :return: The network's output in the shape of [B, C] as a PyTorch tensor that
            - B is an integer number that indicates the Batch size as number of input samples.
            - C is an integer number that indicates the number of output features.
        """

        bert_output, _ = self.bert(a_net_input)
        classification_out = bert_output[:, 0, :]
        output = self.fc_action(classification_out)
        return output

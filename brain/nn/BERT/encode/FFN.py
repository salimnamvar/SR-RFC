""" BERT - Position-wise Feedforward Network

"""


# region Imported Dependencies
import math
import torch
from torch import nn, Tensor
# endregion Imported Dependencies


class PositionwiseFeedForward(nn.Module):
    """ Position-wise Feed Forward Network

        This class is used to define Position-wise Feed Forward block to apply on all positions separately and
        identically. PFFN(x) = W2GELU(W1x + b1) + b2 where GELU(·) is the Gaussian Error Linear Unit (GELU)
        activation function.
    """
    def __init__(self, a_d_model: int, a_d_ff: int, a_dropout: float = 0.1) -> None:
        """ Constructor

        :param a_d_model: An integer that specifies the hidden size of transformer.
        :param a_d_ff: An integer that specifies the feed forward hidden size. It is usually 4 * hidden_size.
        :param a_dropout: A float that specifies the dropout rate.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(a_d_model, a_d_ff)
        self.w_2 = nn.Linear(a_d_ff, a_d_model)
        self.dropout = nn.Dropout(a_dropout)
        self.activation = GELU()

    def forward(self, a_x: Tensor) -> Tensor:
        """ Feedforward

        :param a_x: A tensor that specifies the input data of the layer.
        :return: The layer's output.
        """
        return self.w_2(self.dropout(self.activation(self.w_1(a_x))))


class GELU(nn.Module):
    """ BERT - Gaussian Error Linear Unit (GELU) activation function

        This class is used to define GELU activation function for Position-wise Feed Forward Network.
    """
    def forward(self, a_x: Tensor) -> Tensor:
        """ Feedforward

        :param a_x: A tensor that specifies the input data of the layer.
        :return: The layer's output.
        """
        return 0.5 * a_x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (a_x + 0.044715 * torch.pow(a_x, 3))))

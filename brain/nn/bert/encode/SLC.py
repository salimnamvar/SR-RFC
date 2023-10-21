""" BERTv0 - Sublayer Connection

"""


# region Imported Dependencies
import torch
from torch import nn, Tensor
# endregion Imported Dependencies


class SublayerConnection(nn.Module):
    """ BERTv0 - Sublayer Connection

        This class is used to define sublayer connection that is a residual connection followed by a layer norm. It
        is used in the TransformerBlock of BERTv0.
    """
    def __init__(self, a_size: int, a_dropout: float) -> None:
        """ Constructor

        :param a_size: An integer that specifies the hidden size of transformer.
        :param a_dropout: A float that specifies the dropout rate.
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(a_size)
        self.dropout = nn.Dropout(a_dropout)
    # endregion Constructor

    def forward(self, a_x: Tensor, a_sublayer: nn.Module) -> Tensor:
        """ Feedforward

            It is the main method for running the SublayerConnection module. It is used to apply residual connection
            to any sublayer with the same size.
        :param a_x: A tensor that specifies the input data of the layer.
        :param a_sublayer: A layer for applying the residual connection to.
        :return: The layer's output.
        """
        return a_x + self.dropout(a_sublayer(self.norm(a_x)))


class LayerNorm(nn.Module):
    """ BERTv0 - Normalization Layer

    """
    def __init__(self, a_features: int, a_eps: float = 1e-6) -> None:
        """ Constructor

        :param a_features: An integer that specifies the hidden size of transformer.
        :param a_eps: A small float value as epsilon.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(a_features))
        self.b_2 = nn.Parameter(torch.zeros(a_features))
        self.eps = a_eps

    def forward(self, a_x: Tensor) -> Tensor:
        """ Feedforward

        :param a_x: A tensor that specifies the input data of the layer.
        :return: The layer's output.
        """
        mean = a_x.mean(-1, keepdim=True)
        std = a_x.std(-1, keepdim=True)
        return self.a_2 * (a_x - mean) / (std + self.eps) + self.b_2

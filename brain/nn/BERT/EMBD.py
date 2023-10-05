""" BERT Embedding Layers

"""


# region IMPORT LIBRARY
import torch
from torch import nn, Tensor
# endregion IMPORT LIBRARY


class BERTEmbedding(nn.Module):
    """ BERT - Embedding

        This class is used to define a BERT Embedding which is consisted with under features:
            1. PositionalEmbedding : adding positional information using sin, cos.
            2. Sum of all these features are output of BERTEmbedding.
    """

    def __init__(self, a_input_dim: int, a_max_len: int, a_dropout: float = 0.1) -> None:
        """ Constructor

        :param a_input_dim: An integer that specifies the vocab size of total words.
        :param a_max_len: An integer that specifies embedding size of token embedding.
        :param a_dropout: A float that specifies the dropout rate.
        """
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding(a_d_model=a_input_dim, a_max_len=a_max_len)
        self.dropout = nn.Dropout(p=a_dropout)

    def forward(self, a_sequence: Tensor) -> Tensor:
        """ Feedforward

        :param a_sequence: A tensor that specifies the input data of the layer.
        :return:
        """
        x = self.learnedPosition(a_sequence) + a_sequence
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """ BERT - Learned Positional Embedding

        This class is used to define a Learned Positional Embedding for BERT Embedding.
    """

    def __init__(self, a_d_model: int, a_max_len: int = 512) -> None:
        """ Constructor

        :param a_d_model: An integer that specifies the vocab size of total words.
        :param a_max_len: An integer that specifies embedding size of token embedding.
        """
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(a_max_len, a_d_model).float()
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe)
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, a_x: Tensor) -> Tensor:
        """ Feedforward

        :param a_x: A tensor that specifies the input data of the layer.
        :return: The layer's output.
        """
        return self.pe[:, :a_x.size(1)]
""" BERTv0 - Multi-Head Attention Block

"""


# region Imported Dependencies
import math
import types
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
# endregion Imported Dependencies


class MultiHeadedAttention(nn.Module):
    """ BERTv0 - Multi-headed Attention

        This class is used to define a Multi Heads Attention Module of the BERTv0 model. Instead of performing a single
        attention function with d-model-dimensional keys, values and queries, we found it beneficial to linearly project
        the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions,
        respectively. On each of these projected versions of queries, keys and values we then perform the attention
        function in parallel, yielding dv-dimensional output values. These are concatenated and once again projected,
        resulting in the final values. Multi-head attention allows the model to jointly attend to information from
        different representation subspaces at different positions. With a single attention head, averaging inhibits
        this.
    """

    def __init__(self, a_h: int, a_d_model: int, a_dropout: float = 0.1) -> None:
        """ Constructor

        :param a_h: An integer that specifies the head size of multi-head attention.
        :param a_d_model: An integer that specifies the hidden size of transformer in the BERTv0 model.
        :param a_dropout: A float that specifies the dropout rate.
        """
        super().__init__()
        assert a_d_model % a_h == 0

        # We assume d_v always equals d_k
        self.d_k = a_d_model // a_h
        self.h = a_h

        self.linear_layers = nn.ModuleList([nn.Linear(a_d_model, a_d_model) for _ in range(3)])
        self.output_linear = nn.Linear(a_d_model, a_d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=a_dropout)

    def forward(self, a_query: Tensor, a_key: Tensor, a_value: Tensor, a_mask: bool = None) -> Tensor:
        """ Feedforward

        :param a_query: A tensor that specifies the set of input queries for attention blocks.
        :param a_key: A tensor that specifies the set of input keys for attention blocks.
        :param a_value: A tensor that specifies the set of input values for attention blocks.
        :param a_mask: A mask boolean tensor for applying a mask on the scores before the Softmax.
        :return: The layer's output.
        """
        batch_size = a_query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        a_query, a_key, a_value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                                   for l, x in zip(self.linear_layers, (a_query, a_key, a_value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(a_query, a_key, a_value, a_mask=a_mask, a_dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class Attention(nn.Module):
    """ BERTv0 - Attention Module

        This class is used to define one Attention Module of the BERTv0 model. It computes Scaled Dot Product Attention.
        An attention function can be described as mapping a query and a set of key-value pairs to an output, where
        the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values,
        where the weight assigned to each value is computed by a compatibility function of the query with the
        corresponding key.
        We call our particular attention "Scaled Dot-Product Attention_v0". The input consists of queries and keys of
        dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each
        by √dk, and apply a softmax function to obtain the weights on the values.
    """

    def forward(self, a_query: Tensor, a_key: Tensor, a_value: Tensor, a_mask: bool = None,
                a_dropout: types.FunctionType = None) -> Tuple[Tensor, Tensor]:
        """ Feedforward

        :param a_query: A tensor that specifies the set of input queries.
        :param a_key: A tensor that specifies the set of input keys.
        :param a_value: A tensor that specifies the set of input values.
        :param a_mask: A mask boolean tensor for applying a mask on the scores before the Softmax.
        :param a_dropout: A callable function object that is used as the dropout module to apply dropout on the Attention.
        :return: The layer's output.
        """
        scores = torch.matmul(a_query, a_key.transpose(-2, -1)) / math.sqrt(a_query.size(-1))
        scores = scores.to('cuda')

        if a_mask is not None:
            _MASKING_VALUE = torch.tensor(-1e9).to('cuda') if scores.dtype == torch.float32 else torch.tensor(-1e4).to('cuda')
            scores = scores.masked_fill(a_mask.to('cuda') == torch.tensor(0), _MASKING_VALUE)

        p_attn = F.softmax(scores, dim=-1)

        if a_dropout is not None:
            p_attn = a_dropout(p_attn)

        return torch.matmul(p_attn, a_value), p_attn
""" BERTv0 - Transformer Block

"""


# region Imported Dependencies
from torch import nn, Tensor
from brain.nn.BERTv0.encode.FFN import PositionwiseFeedForward
from brain.nn.BERTv0.encode.MHT import MultiHeadedAttention
from brain.nn.BERTv0.encode.SLC import SublayerConnection
# endregion Imported Dependencies


class TransformerBlock(nn.Module):
    """ BERTv0 - Transformer Block

        This class is used to define a Bidirectional Encoder that is equal to Transformer (self-attention).
        Transformer = MultiHead_Attention + Feed_Forward with sublayer connection.
    """

    def __init__(self, a_hidden: int, a_attn_heads: int, a_feed_forward_hidden: int, a_dropout: float) -> None:
        """ Constructor

        :param a_hidden: An integer that specifies the hidden size of transformer.
        :param a_attn_heads: An integer that specifies the head size of multi-head attention.
        :param a_feed_forward_hidden: An integer that specifies the feed forward hidden size.
        It is usually 4 * hidden_size.
        :param a_dropout: A float that specifies the dropout rate.
        """
        super().__init__()
        self.attention = MultiHeadedAttention(a_h=a_attn_heads, a_d_model=a_hidden)
        self.feed_forward = PositionwiseFeedForward(a_d_model=a_hidden, a_d_ff=a_feed_forward_hidden,
                                                    a_dropout=a_dropout)
        self.input_sublayer = SublayerConnection(a_size=a_hidden, a_dropout=a_dropout)
        self.output_sublayer = SublayerConnection(a_size=a_hidden, a_dropout=a_dropout)
        self.dropout = nn.Dropout(p=a_dropout)

    def forward(self, a_x, a_mask) -> Tensor:
        """ Feedforward

        :param a_x: A tensor that specifies the input data of the layer.
        :param a_mask: A mask boolean tensor for applying a mask on the scores in the attention blocks.
        :return: The layer's output.
        """
        a_x = self.input_sublayer(a_x, lambda _x: self.attention.forward(_x, _x, _x, a_mask=a_mask))
        a_x = self.output_sublayer(a_x, self.feed_forward)
        return self.dropout(a_x)
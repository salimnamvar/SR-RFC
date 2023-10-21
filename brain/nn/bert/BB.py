""" BERTv0 Backbone

"""

# region Imported Dependencies
from typing import Tuple
import torch
from torch import nn, Tensor
from brain.nn.bert.EMBD import BERTEmbedding
from brain.nn.bert.encode.TNet import TransformerBlock
# endregion Imported Dependencies


class BERT(nn.Module):
    """ BERTv0 Model

        This class is used to define a Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, a_input_dim: int, a_max_len: int, a_hidden: int = 768, a_n_layers: int = 12,
                 a_attn_heads: int = 12, a_dropout: float = 0.1, a_mask_prob: float = 0.8):
        """ Constructor

        :param a_input_dim: An integer that specifies the vocab size of total words.
        :param a_max_len: An integer that specifies embedding size of token embedding.
        :param a_hidden: An integer that specifies the BERTv0 model hidden size.
        :param a_n_layers: An integer that specifies the number of Transformer blocks(layers).
        :param a_attn_heads: An integer that specifies the number of attention heads.
        :param a_dropout: A float that specifies the dropout rate.
        :param a_mask_prob: A float that specifies the mask probability value in the Bernoulli Distribution.
        """
        super().__init__()
        self.hidden = a_hidden
        self.n_layers = a_n_layers
        self.attn_heads = a_attn_heads
        self.max_len = a_max_len
        self.input_dim = a_input_dim
        self.mask_prob = a_mask_prob

        clsToken = torch.zeros(1, 1, self.input_dim).float()
        clsToken.require_grad = True
        self.clsToken = nn.Parameter(clsToken)
        torch.nn.init.normal_(clsToken, std=0.02)

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = a_hidden * 4

        # embedding for BERTv0, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(a_input_dim=a_input_dim, a_max_len=a_max_len + 1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(a_hidden, a_attn_heads, self.feed_forward_hidden, a_dropout) for _ in range(a_n_layers)])

    def forward(self, a_input_vectors: Tensor) -> Tuple[Tensor, Tensor]:
        """ Feedforward

        :param a_input_vectors: A tensor that specifies the input vector of the layer.
        :return: The layer's output
        """

        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size = a_input_vectors.shape[0]
        sample = None
        if self.training:
            bernolliMatrix = torch.cat((torch.tensor([1]).float(),
                                        (torch.tensor([self.mask_prob]).float()).repeat(self.max_len)),
                                       0).unsqueeze(0).repeat([batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len + 1, self.max_len + 1)

        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size, 1, 1), a_input_vectors), 1)
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x, sample

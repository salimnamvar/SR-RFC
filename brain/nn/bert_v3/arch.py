""" Transformers BERT - Architecture 3

"""

# region Imported Dependencies
import torch
from torch import nn
from transformers import BertModel


# endregion Imported Dependencies


class Arch(nn.Module):
    def __init__(self, a_max_length: int = 457) -> None:
        super().__init__()
        self.bert_layer: BertModel = BertModel.from_pretrained('bert-base-uncased')
        self.dropout_layer: nn.Dropout = nn.Dropout(0.3)
        self.reactivity_layer1: nn.Linear = nn.Linear(768, a_max_length)
        self.reactivity_layer2: nn.Linear = nn.Linear(a_max_length, a_max_length)
        self.reactivity_layer3: nn.Linear = nn.Linear(a_max_length, a_max_length)
        self.sigmoid_layer: nn.Sigmoid = nn.Sigmoid()

    def forward(self, a_input_ids: torch.Tensor, a_attention_mask: torch.Tensor,
                a_token_type_ids: torch.Tensor) -> torch.Tensor:
        _, x = self.bert_layer(input_ids=a_input_ids, attention_mask=a_attention_mask,
                               token_type_ids=a_token_type_ids, return_dict=False)
        x = self.dropout_layer(x)
        x = self.reactivity_layer1(x)
        return x

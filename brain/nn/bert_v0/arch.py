""" Transformers BERT - Architecture 0

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
        self.dropout_layer: nn.Dropout = nn.Dropout(0.1)
        self.reactivity_layer: nn.Linear = nn.Linear(768, a_max_length)
        self.sigmoid_layer: nn.Sigmoid = nn.Sigmoid()

    def forward(self, a_input_ids: torch.Tensor, a_attention_mask: torch.Tensor,
                a_token_type_ids: torch.Tensor) -> torch.Tensor:
        _, output1 = self.bert_layer(input_ids=a_input_ids, attention_mask=a_attention_mask,
                                     token_type_ids=a_token_type_ids, return_dict=False)
        output2 = self.dropout_layer(output1)
        output = self.reactivity_layer(output2)
        return output

    def test(self, a_input_ids: torch.Tensor, a_attention_mask: torch.Tensor,
             a_token_type_ids: torch.Tensor) -> torch.Tensor:
        output = self.forward(a_input_ids, a_attention_mask, a_token_type_ids)
        output_sig = self.sigmoid_layer(output)
        return output_sig

""" Transformers BERT - Architecture 0

"""


# region Imported Dependencies
from torch import nn
from transformers import BertModel
# endregion Imported Dependencies


class Bert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert_layer: BertModel = BertModel.from_pretrained('bert-base-uncased')
        self.dropout_layer: nn.Dropout = nn.Dropout(0.3)
        self.reactivity_layer: nn.Linear = nn.Linear(768, 457)

    def forward(self, a_input_ids, a_attention_mask, a_token_type_ids):
        _, output1 = self.bert_layer(input_ids=a_input_ids, attention_mask=a_attention_mask,
                                     token_type_ids=a_token_type_ids, return_dict=False)
        output2 = self.dropout_layer(output1)
        output = self.reactivity_layer(output2)
        return output

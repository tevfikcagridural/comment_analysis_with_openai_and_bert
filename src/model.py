import torch
import transformers

class BERTClass(torch.nn.Module):
    def __init__(self, is_aspect: bool=True):
        # There are 5 aspects and 2 classes
        if is_aspect:
            output_shape = 5
        else:
            output_shape = 2
        super(BERTClass, self).__init__()
        self.base = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.classification = torch.nn.Linear(768, output_shape)

    def forward(self, ids, mask, token_type_ids):
        _, X= self.base(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        X = self.dropout(X)
        output = self.classification(X)
        return output
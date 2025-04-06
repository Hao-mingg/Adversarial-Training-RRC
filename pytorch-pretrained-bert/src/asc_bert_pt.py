import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class BertForABSA(BertPreTrainedModel):
    def __init__(self, config, num_labels=3, dropout=0.1):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # Initialize weights (Hugging Face handles this internally)
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True  # Modern Hugging Face practice
        )
        pooled_output = outputs.pooler_output  # Extract [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits

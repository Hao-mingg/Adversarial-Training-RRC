import torch
import torch.nn as nn
from torch.autograd import grad
from transformers import BertPreTrainedModel, BertModel

class BertForABSA(BertPreTrainedModel):
    def __init__(self, config, num_labels=3, dropout=0.1, epsilon=1.0):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.epsilon = epsilon  # Perturbation factor for adversarial training
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # Initialize weights (Handled internally by Hugging Face)
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True  # Returns a structured dictionary
        )
        pooled_output = outputs.pooler_output  # Extract the [CLS] representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # Apply adversarial training if the model is in training mode
            if self.training and pooled_output.requires_grad:
                perturbed_emb = self.adv_attack(outputs.last_hidden_state, loss, self.epsilon)
                adv_loss = self.adversarial_loss(perturbed_emb, attention_mask, labels)
                return loss, adv_loss

            return loss
        return logits

    def adv_attack(self, emb, loss, epsilon):
        """
        Generates adversarial perturbations in the embedding space.
        """
        loss_grad = grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.norm(loss_grad, dim=(1, 2), keepdim=True)
        perturbed_emb = emb + epsilon * (loss_grad / (loss_grad_norm + 1e-8))  # Adding small constant for stability
        return perturbed_emb

    def adversarial_loss(self, perturbed_emb, attention_mask, labels):
        """
        Computes adversarial loss using the perturbed embeddings.
        """
        outputs = self.bert(
            inputs_embeds=perturbed_emb,  # Uses perturbed embeddings instead of input IDs
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        adv_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return adv_loss

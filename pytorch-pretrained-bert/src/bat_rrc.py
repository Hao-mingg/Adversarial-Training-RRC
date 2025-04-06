import torch
import torch.nn as nn
from torch.autograd import grad
from transformers import BertForQuestionAnswering, BertModel, BertPreTrainedModel

class BertForQAWithAdversarialTraining(BertForQuestionAnswering):
    def __init__(self, config, epsilon=1.0, dropout=0.1):
        super(BertForQAWithAdversarialTraining, self).__init__(config)
        self.epsilon = epsilon  # Perturbation strength
        self.dropout = nn.Dropout(dropout)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_positions=None, end_positions=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True
        )

        logits_start, logits_end = outputs.start_logits, outputs.end_logits
        hidden_states = outputs.hidden_states[-1]
        if start_positions is not None and end_positions is not None:
            start_loss = self.loss_fct(logits_start, start_positions)
            end_loss = self.loss_fct(logits_end, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if self.training and hidden_states.requires_grad:
                perturbed_emb = self.adv_attack(hidden_states, total_loss)
                perturbed_emb = self.replace_cls_token(hidden_states, perturbed_emb)

                adv_loss = self.adversarial_loss(perturbed_emb, attention_mask, token_type_ids, start_positions, end_positions)
                return total_loss, adv_loss

            return total_loss
        return logits_start, logits_end

    def adv_attack(self, emb, loss):
        loss_grad = grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.norm(loss_grad, dim=(1, 2), keepdim=True)
        perturbed_emb = emb + self.epsilon * (loss_grad / (loss_grad_norm + 1e-8))  # Prevents division by zero
        return perturbed_emb

    def replace_cls_token(self, emb, perturbed):
        """
        Ensures that the [CLS] token remains unchanged during adversarial perturbation.
        """
        cond = torch.zeros_like(emb, dtype=torch.bool)
        cond[:, 0, :] = True  # Keep [CLS] token unchanged
        perturbed_sentence = torch.where(cond, emb, perturbed)
        return perturbed_sentence

    def adversarial_loss(self, perturbed_emb, attention_mask, token_type_ids, start_positions, end_positions):
        """
        Computes adversarial loss using the perturbed embeddings.
        """
        outputs = self.bert(
            inputs_embeds=perturbed_emb,  # Uses perturbed embeddings instead of input IDs
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        logits_start, logits_end = self.qa_outputs(outputs.last_hidden_state).split(1, dim=-1)
        logits_start, logits_end = logits_start.squeeze(-1), logits_end.squeeze(-1)

        start_loss = self.loss_fct(logits_start, start_positions)
        end_loss = self.loss_fct(logits_end, end_positions)
        adv_loss = (start_loss + end_loss) / 2
        return adv_loss

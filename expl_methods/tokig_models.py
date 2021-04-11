from transformers import RobertaForQuestionAnswering
from transformers.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaEncoder,
    RobertaAttention,
    RobertaSelfAttention,
    RobertaEmbeddings,
    RobertaAttention,
    RobertaSelfOutput,
    RobertaIntermediate,
    RobertaOutput,
    RobertaLayer,
    create_position_ids_from_input_ids,
)

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
import torch
from torch import nn
import torch.nn.functional as F
import math


class TokIGRobertaForQuestionAnswering(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)        
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        do_attribute=False,
        **kwargs,
    ):
        if do_attribute:
            return self.attribute(**kwargs)
        else:
            return self.predict(**kwargs)

    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _get_origin_emb(self, input_ids):
        return self.roberta.embeddings.word_embeddings(input_ids)
    
    def _get_origin_position_ids(self, input_ids):
        return create_position_ids_from_input_ids(input_ids, self.config.pad_token_id).to(input_ids.device)

    @staticmethod
    def probs_of_span(start_logits, end_logits, start_indexes, end_indexes):
        start_logits = F.softmax(start_logits, dim=1)
        end_logits = F.softmax(end_logits, dim=1)
        selected_start_logits = torch.gather(start_logits, 1, start_indexes.view(-1, 1))
        selected_end_logits = torch.gather(end_logits, 1, end_indexes.view(-1, 1))
        sum_logits = selected_start_logits * selected_end_logits

        return sum_logits

    def attribute(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,       
        start_indexes=None,
        end_indexes=None,
        final_start_logits=None, # for comparison
        final_end_logits=None, # for comparison
        num_steps=20
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        acc_attribution = .0                 
        # input size B * L
        # attention mask B * L
        # N_Layer * B * N_HEAD * L * L
        # TODO: FIX hard encoding
        mask_token_id = 50264
        baseline_input_ids = mask_token_id * torch.ones_like(input_ids)        
        baseline_embs = self._get_origin_emb(baseline_input_ids)
        target_embs = self._get_origin_emb(input_ids)
        target_position_ids = self._get_origin_position_ids(input_ids)
        diff_embs = target_embs - baseline_embs        
        for step_i in range(num_steps):    
            # compose input
            step_embs = diff_embs * step_i / num_steps + baseline_embs
            step_embs.requires_grad_(True)
            
            outputs = self.roberta(
                input_ids=None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=target_position_ids,
                head_mask=head_mask,
                inputs_embeds=step_embs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]

            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
    
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            pred_probs = self.probs_of_span(start_logits, end_logits, start_indexes, end_indexes)
            step_loss = torch.sum(pred_probs)
            step_loss.backward()

            step_attribution = step_embs.grad.detach() * diff_embs / num_steps
            acc_attribution += step_attribution

            if step_i == 0:
                baseline_sum_logits = pred_probs
        

        # sanity check
        # N_Layer * B * N_HEAD * L * L
        # final_sum_logits = self.probs_of_span(final_start_logits, final_end_logits, start_indexes, end_indexes)
        # diff_sum_logits = final_sum_logits - baseline_sum_logits
        # sum_attribution = torch.sum(acc_attribution, (2,1))
        # print(baseline_sum_logits.view([-1]))
        # print(final_sum_logits.view([-1]))
        # print(diff_sum_logits.view([-1]))
        # print(sum_attribution.view([-1]))

        acc_attribution = torch.sum(acc_attribution, 2)
        return acc_attribution
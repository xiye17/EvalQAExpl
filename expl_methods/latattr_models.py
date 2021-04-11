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
    RobertaLayer
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

# Copied from transformers.modeling_bert.BertSelfAttention with Bert->Roberta
class LAtAttrRobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # possible probation
    # disable a layer, i.e., zeroing out all the links
    # mask a layer, i.e., mask some attention
    # specify attention
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        input_attentions=None,
        link_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)


        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)


        if input_attentions is None:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
        else:
            attention_probs = input_attentions

        if link_mask is not None:
            attention_probs = attention_probs * link_mask
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

# Copied from transformers.modeling_bert.BertAttention with Bert->Roberta
class LAtAttrRobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LAtAttrRobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        input_attentions=None,
        link_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            input_attentions,
            link_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.modeling_bert.BertLayer with Bert->Roberta
class LAtAttrRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LAtAttrRobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        input_attentions=None,
        link_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            input_attentions=input_attentions,
            link_mask=link_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class LAtAttrRobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LAtAttrRobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        input_attentions=None,
        link_mask=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_input_attentions = input_attentions[i] if input_attentions is not None else None
            layer_link_mask = link_mask[i] if link_mask is not None else None
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, layer_input_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    layer_input_attentions,
                    layer_link_mask
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

class LAtAttrRobertaModel(RobertaPreTrainedModel):


    authorized_missing_keys = [r"position_ids"]

    # Copied from transformers.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = LAtAttrRobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):

        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_attentions=None,
        link_mask=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_attentions=input_attentions,
            link_mask=link_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class LAtAttrRobertaForQuestionAnswering(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = LAtAttrRobertaModel(config, add_pooling_layer=False)        
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_attention_heads = self.config.num_attention_heads


    def forward(
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
        input_attentions=None,
        link_mask=None,
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
            input_attentions=input_attentions,
            link_mask=link_mask,
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

    @staticmethod
    def probs_of_span(start_logits, end_logits, start_indexes, end_indexes):
        start_logits = F.softmax(start_logits, dim=1)
        end_logits = F.softmax(end_logits, dim=1)
        selected_start_logits = torch.gather(start_logits, 1, start_indexes.view(-1, 1))
        selected_end_logits = torch.gather(end_logits, 1, end_indexes.view(-1, 1))
        sum_logits = selected_start_logits * selected_end_logits

        return sum_logits    

    # batch size has to be 1
    def probe_forward(
        self,
        input_ids=None,
        start_indexes=None,
        end_indexes=None,
        final_start_logits=None, # for comparison
        final_end_logits=None, # for comparison
        active_layers=None,
        link_mask=None,
        return_kl=True,
        **kwargs
    ):

        if active_layers is None:
            input_attentions = None
        else:
            zero_input_attention = torch.zeros(
                        [input_ids.size(0), self.config.num_attention_heads, input_ids.size(1), input_ids.size(1)],
                        device=input_ids.device)
            input_attentions = [ None if active_layers[i] else zero_input_attention for i in range(self.num_hidden_layers)]
            # in_link_masks = []
            # identity_matrix = torch.eye(input_ids.size(1), dtype=torch.bool, device=input_ids.device)
            # identity_matrix = identity_matrix.expand(1, self.config.num_attention_heads, -1, -1)
            # for i in range(self.num_hidden_layers):
            #     if link_mask is not None and link_mask[i] is not None:
            #         in_link_masks.append(link_mask[i])
            #     else:
            #         in_link_masks.append(None if active_layers[i] else identity_matrix)            

        batch_start_logits, batch_end_logits = self.forward(input_ids=input_ids, input_attentions=input_attentions, link_mask=link_mask, **kwargs)
        batch_probs = self.probs_of_span(batch_start_logits, batch_end_logits, start_indexes, end_indexes)
        if not return_kl:
            return batch_probs.item()

        kl_loss = F.kl_div(
            F.log_softmax(batch_start_logits, dim=1),
            F.softmax(final_start_logits, dim=1),
            reduction='batchmean',
        ) + F.kl_div(
            F.log_softmax(batch_end_logits, dim=1),
            F.softmax(final_end_logits, dim=1),
            reduction='batchmean'                
        )
        return batch_probs.item(), kl_loss.item()

    def restricted_forward(
        self,
        input_ids=None,
        active_layers=None,
        link_mask=None,
        **kwargs,
    ):
        if active_layers is None:
            input_attentions = None
        else:
            zero_input_attention = torch.zeros(
                        [input_ids.size(0), self.config.num_attention_heads, input_ids.size(1), input_ids.size(1)],
                        device=input_ids.device)
            input_attentions = [ None if active_layers[i] else zero_input_attention for i in range(self.num_hidden_layers)]

        return self.forward(input_ids=input_ids, input_attentions=input_attentions, link_mask=link_mask, **kwargs)

    def layer_attribute(
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
        active_layers=None,
        link_mask=None,
        input_attentions=None,        
        start_indexes=None,
        end_indexes=None,
        final_start_logits=None, # for comparison
        final_end_logits=None, # for comparison
        num_steps=300,
    ):
        # for parallel        
        # input size B * L
        # attention mask B * L
        # N_Layer * B * N_HEAD * L * L

        # # sanity check
        # N_Layer * B * N_HEAD * L * L
        final_probs = self.probs_of_span(final_end_logits, final_end_logits, start_indexes, end_indexes)

        attributions = []
        
        if active_layers is None:
            active_layers = [1 for _ in range(self.num_hidden_layers)]
        zero_input_attention = torch.zeros(
                    [input_ids.size(0), self.config.num_attention_heads, input_ids.size(1), input_ids.size(1)],
                    device=input_ids.device)
        

        for layer_i in range(self.num_hidden_layers):

            if not active_layers[layer_i]:
                attributions.append(zero_input_attention)
                continue

            layer_acc_attribution = .0
            layer_input_attention_backbone = [ None if active_layers[i] else zero_input_attention for i in range(self.num_hidden_layers)]
            layer_attention = input_attentions[layer_i]

            for step_i in range(num_steps):            
                # compose input
                step_attentions = layer_attention * step_i / num_steps
                step_attentions.requires_grad_(True)
                layer_input_attention_backbone[layer_i] = step_attentions

                outputs = self.roberta(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    input_attentions=layer_input_attention_backbone,
                    link_mask=link_mask,
                )

                sequence_output = outputs[0]

                logits = self.qa_outputs(sequence_output)
                start_logits, end_logits = logits.split(1, dim=-1)
        
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)

                span_probs = self.probs_of_span(start_logits, end_logits, start_indexes, end_indexes)
                step_loss = torch.sum(span_probs)
                step_loss.backward()

                step_attribution = step_attentions.grad.detach() * layer_attention / num_steps
                layer_acc_attribution += step_attribution

                if step_i == 0:
                    baseline_probs = span_probs
            # print(layer_acc_attribution.size())
            sum_attribution = torch.sum(layer_acc_attribution, (3,2,1))
            sanity_probs = baseline_probs + sum_attribution
            # print(layer_i, final_probs.item(), sanity_probs.item(), baseline_probs.item(),  sum_attribution.item())
            attributions.append(layer_acc_attribution)
        attributions = torch.stack(attributions)
        # print(attributions.size())
        return attributions


class AtAttrRobertaForQuestionAnswering(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = LAtAttrRobertaModel(config, add_pooling_layer=False)        
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
        input_attentions=None,
        start_indexes=None,
        end_indexes=None,
        final_start_logits=None, # for comparison
        final_end_logits=None, # for comparison
        num_steps=50,
        is_parallel=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        acc_attribution = .0 
        # for parallel
        if is_parallel:
            input_attentions = torch.transpose(input_attentions, 0, 1)
        # input size B * L
        # attention mask B * L
        # N_Layer * B * N_HEAD * L * L
                
        for step_i in range(num_steps):            
            # compose input
            step_attentions = input_attentions * step_i / num_steps
            step_attentions.requires_grad_(True)
            
            
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
                input_attentions=step_attentions,
            )

            sequence_output = outputs[0]

            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
    
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            pred_probs = LAtAttrRobertaForQuestionAnswering.probs_of_span(start_logits, end_logits, start_indexes, end_indexes)
            step_loss = torch.sum(pred_probs)
            step_loss.backward()

            step_attribution = step_attentions.grad.detach() * input_attentions / num_steps
            acc_attribution += step_attribution

            if step_i == 0:
                baseline_sum_logits = pred_probs
        

        # # sanity check
        # N_Layer * B * N_HEAD * L * L
        # final_sum_logits = LAtAttrRobertaForQuestionAnswering.probs_of_span(final_start_logits, final_end_logits, start_indexes, end_indexes)
        # diff_sum_logits = final_sum_logits - baseline_sum_logits
        # sum_attribution = torch.sum(acc_attribution, (4,3,2,0))
        # print(baseline_sum_logits.view([-1]))
        # print(final_sum_logits.view([-1]))
        # print(diff_sum_logits.view([-1]))
        # print(sum_attribution.view([-1]))

        if is_parallel:
            acc_attribution = torch.transpose(acc_attribution, 0, 1)
        return acc_attribution

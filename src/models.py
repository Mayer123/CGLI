import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from roberta import RobertaModel
from transformers import RobertaModel as OriginalRobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator, sumed_scores = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum(), sumed_scores

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        sumed_scores = [score]
        max_score = score
        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)

            next_max_score, _ = next_score.max(dim=1)
            next_score = torch.logsumexp(next_score, dim=1)
            
            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            max_score = torch.where(mask[i].unsqueeze(1), next_max_score, max_score)
            sumed_scores.append(max_score)
        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions
        sumed_scores[-1] = score
        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1), sumed_scores

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        sumed_scores = [score]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            sumed_scores.append(score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions
        sumed_scores[-1] = score
        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list, sumed_scores

class RobertaProPara(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, num_states):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.CRFLayer = CRF(num_states, batch_first = True)
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, num_states)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        timestep_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        state_change_labels=None,
        test=False,
        do_mml=False,
        num_steps=-1
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            timestep_type_ids=timestep_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)     
        start_logits, end_logits = logits.split(1, dim=-1)
        cls_token = sequence_output[:, 0, :]
        cls_token = cls_token.unsqueeze(0)
        change_rep, state_changes = self.get_change_rep_action(cls_token)


        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loc_loss = None
        if (start_positions is not None and end_positions is not None):
            if not do_mml:
                start_positions = start_positions[:, 0]
                end_positions = end_positions[:, 0]

                loss_fct = CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loc_loss = (start_loss + end_loss) / 2

        ent_loss = None
        if state_change_labels is not None:
            state_change_labels = state_change_labels.unsqueeze(0)
            # The state changes with shape (num_entities, num_steps-1, num_tags)
            log_likelihood, sumed_scores = self.CRFLayer(emissions = state_changes, tags = state_change_labels, reduction = 'token_mean')
            ent_loss = -log_likelihood 
           
        return QuestionAnsweringModelOutput(
            loss=loc_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), (loc_loss, ent_loss), state_changes

    def get_change_rep_action(self, cls_token):
        before = cls_token[:, :-1, :]
        after = cls_token[:, 1:, :]
        change_rep = torch.cat([before, after], dim=-1)
        change_rep = self.dropout(change_rep)
        state_changes = self.out_proj(self.dropout(torch.tanh(self.dense(change_rep))))
        return change_rep, state_changes


class RobertaTripNoCRF(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        all_dense_pre = []
        all_dense_effect = []
        all_out_pre = []
        all_out_effect = []
        for i in range(20):
            dense_pre = nn.Linear(config.hidden_size, config.hidden_size)
            dense_effect = nn.Linear(config.hidden_size, config.hidden_size)
            if i == 5:
                out_pre = nn.Linear(config.hidden_size, 9)
                out_effect = nn.Linear(config.hidden_size, 9)
            else:
                out_pre = nn.Linear(config.hidden_size, 3)
                out_effect = nn.Linear(config.hidden_size, 3)
            all_out_pre.append(out_pre)
            all_out_effect.append(out_effect)
            all_dense_pre.append(dense_pre)
            all_dense_effect.append(dense_effect)
        self.all_out_pre = nn.ModuleList(all_out_pre)
        self.all_out_effect = nn.ModuleList(all_out_effect)
        self.all_dense_pre = nn.ModuleList(all_dense_pre)
        self.all_dense_effect = nn.ModuleList(all_dense_effect)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.conflict_proj = nn.Linear(config.hidden_size*2, 1)

        self.plausible_proj = nn.Linear(config.hidden_size, 2)
        self.loss_fct = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        timestep_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        effect_labels=None, conflict_label=None, plausible_label=None, precondition_labels=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            timestep_type_ids=timestep_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        cls_token = sequence_output[:, 0, :]
        cls_token = cls_token.unsqueeze(0)
       
        change_rep = self.dropout(cls_token)
       
        change_loss = None
        conflict_loss = None
        plausible_loss = None
        
        all_change_pres = []
        all_change_effs = []
        for i in range(20):
            change_pre = self.all_out_pre[i](self.dropout(torch.tanh(self.all_dense_pre[i](change_rep))))
            all_change_pres.append(change_pre)
            change_eff = self.all_out_effect[i](self.dropout(torch.tanh(self.all_dense_effect[i](change_rep))))
            all_change_effs.append(change_eff)

        if effect_labels is not None:
            change_loss = 0
            for i in range(20):
                pre_label = precondition_labels[:, i]
                change_loss += self.loss_fct(all_change_pres[i].squeeze(0), pre_label)
                eff_label = effect_labels[:, i]
                change_loss += self.loss_fct(all_change_effs[i].squeeze(0), eff_label)
            change_loss /= 20

        conflict_reps = []
        for i in range(change_rep.shape[1]):
            for j in range(i+1, change_rep.shape[1]):
                conflict_reps.append(torch.cat([change_rep[0, i], change_rep[0, j]], dim=-1))
        conflict_reps = self.conflict_proj(torch.stack(conflict_reps)).squeeze(-1).unsqueeze(0)
        if conflict_label is not None:
            conflict_loss = self.loss_fct(conflict_reps, conflict_label.unsqueeze(0))
        

        plausible_rep = self.plausible_proj(torch.mean(change_rep, dim=1))
        if plausible_label is not None:
            plausible_loss = self.loss_fct(plausible_rep, plausible_label.unsqueeze(0))    
        return (change_loss, conflict_loss, plausible_loss), (all_change_pres, all_change_effs, conflict_reps, plausible_rep)


class RobertaTripWithCRF(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        all_CRFs_pre = []
        all_CRFs_eff = []
        all_dense_pre = []
        all_dense_effect = []
        all_out_pre = []
        all_out_effect = []
        for i in range(20):
            dense_pre = nn.Linear(config.hidden_size, config.hidden_size)
            dense_effect = nn.Linear(config.hidden_size, config.hidden_size)
            if i == 5:
                out_pre = nn.Linear(config.hidden_size, 9)
                out_effect = nn.Linear(config.hidden_size, 9)
                CRF_pre = CRF(9, batch_first = True)
                CRF_eff = CRF(9, batch_first = True)
            else:
                out_pre = nn.Linear(config.hidden_size, 3)
                out_effect = nn.Linear(config.hidden_size, 3)
                CRF_pre = CRF(3, batch_first = True)
                CRF_eff = CRF(3, batch_first = True)
            all_out_pre.append(out_pre)
            all_out_effect.append(out_effect)
            all_dense_pre.append(dense_pre)
            all_dense_effect.append(dense_effect)
            all_CRFs_pre.append(CRF_pre)
            all_CRFs_eff.append(CRF_eff)
        self.all_CRFs_pre = nn.ModuleList(all_CRFs_pre)
        self.all_CRFs_eff = nn.ModuleList(all_CRFs_eff)
        self.all_out_pre = nn.ModuleList(all_out_pre)
        self.all_out_effect = nn.ModuleList(all_out_effect)
        self.all_dense_pre = nn.ModuleList(all_dense_pre)
        self.all_dense_effect = nn.ModuleList(all_dense_effect)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.conflict_proj = nn.Linear(config.hidden_size*2, 1)

        self.plausible_proj = nn.Linear(config.hidden_size, 2)
        self.loss_fct = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        timestep_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        effect_labels=None, conflict_label=None, plausible_label=None, precondition_labels=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            timestep_type_ids=timestep_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        cls_token = sequence_output[:, 0, :]
        cls_token = cls_token.unsqueeze(0)
       
        change_rep = self.dropout(cls_token)
       
        change_loss = None
        conflict_loss = None
        plausible_loss = None
        
        all_change_pres = []
        all_change_effs = []
        for i in range(20):
            change_pre = self.all_out_pre[i](self.dropout(torch.tanh(self.all_dense_pre[i](change_rep))))
            all_change_pres.append(change_pre)
            change_eff = self.all_out_effect[i](self.dropout(torch.tanh(self.all_dense_effect[i](change_rep))))
            all_change_effs.append(change_eff)

        if effect_labels is not None:
            change_loss = 0
            for i in range(20):
                pre_label = precondition_labels[:, i].unsqueeze(0)
                pre_log_likelihood, _ = self.all_CRFs_pre[i](emissions = all_change_pres[i], tags = pre_label, reduction = 'token_mean')
                change_loss -= pre_log_likelihood 
                eff_label = effect_labels[:, i].unsqueeze(0)
                eff_log_likelihood, _ = self.all_CRFs_eff[i](emissions = all_change_effs[i], tags = eff_label, reduction = 'token_mean')
                change_loss -= eff_log_likelihood 
            change_loss /= 20

        conflict_reps = []
        for i in range(change_rep.shape[1]):
            for j in range(i+1, change_rep.shape[1]):
                conflict_reps.append(torch.cat([change_rep[0, i], change_rep[0, j]], dim=-1))
        conflict_reps = self.conflict_proj(torch.stack(conflict_reps)).squeeze(-1).unsqueeze(0)
        if conflict_label is not None:
            conflict_loss = self.loss_fct(conflict_reps, conflict_label.unsqueeze(0))

        plausible_rep = self.plausible_proj(torch.mean(change_rep, dim=1))
        if plausible_label is not None:
            plausible_loss = self.loss_fct(plausible_rep, plausible_label.unsqueeze(0))    
        return (change_loss, conflict_loss, plausible_loss), (all_change_pres, all_change_effs, conflict_reps, plausible_rep)
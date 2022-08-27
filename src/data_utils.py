import torch
import json 
import csv
import string
from collections import Counter, defaultdict
import math
import numpy as np
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_sm')
swap_dict = {'alveolus': 'alveoli', 'sew machine':'machine', 'recycle center': 'recycling center', 'recycle facility':'recycling facility',
'leaf': 'leaves', 'stamen': 'stamens', 'bacterium':'bacteria', 'recycle bin': 'recycling bin', 'air sac':'air sacs', 'sac':'sacs',
'vent , hot water through baseboard , or by steam radiator':'vents , hot water through baseboards , or by steam radiators .',
'wind , animal , etc. .':'wind , animals , etc', 'recycle container':'recycling containers', 'recycle plant':'recycling plant',
'ebody of water': 'bodies of water', 'green chemical in the leaf':'green chemical in the leaves', 'turn mechanism':'turning mechanism',
'can or bag': 'can or in bags', 'centralize container':'centralized container', 'wash machine': 'washing machine', 'upward':'upwards', 'drop':'droppings'}

def read_propara_data(file):
    with open(file, 'r') as f:
        lines = []
        for line in f:
            if line.strip() == '':
                continue
            lines.append(json.loads(str(line)))
    return lines

def compute_priors():
    state2idx = {'O_C': 0, 'O_D': 1, 'E': 2, 'M': 3, 'C': 4, 'D': 5}
    start_transitions = Counter()
    end_transitions = Counter()
    transitions = defaultdict(Counter)
    data = read_propara_data('../data/grids.v1.train.json')
    for sample in data:
        for entity_id, states in enumerate(sample['states']):
            state_transition = compute_state_change_seq(states)
            start_transitions[state_transition[0]] += 1
            end_transitions[state_transition[-1]] += 1
            for i in range(1, len(state_transition)):
                transitions[state_transition[i-1]][state_transition[i]] += 1
    
    start_prior = np.full(6, -1e9)
    end_prior = np.full(6, -1e9)
    transition_prior = np.full((6, 6), -1e9)
    for k, v in start_transitions.items():
        start_prior[state2idx[k]] = math.log(v/sum(start_transitions.values()))
    for k, v in end_transitions.items():
        end_prior[state2idx[k]] = math.log(v/sum(end_transitions.values()))
    for k, v in transitions.items():
        for kk, vv in v.items():
            transition_prior[state2idx[k]][state2idx[kk]] = math.log(vv/sum(v.values()))
    return start_prior, end_prior, transition_prior

def check_loc(loc_tokens, para_tokens):
    for i in range(len(para_tokens)):
        if para_tokens[i:i+len(loc_tokens)] == loc_tokens:
            return True
    return False

def make_timestep_sequence(question_tokens, sentence_tokens):
    total_len = 2+len(question_tokens) + sum([len(sent) for sent in sentence_tokens])
    timestep_ids = []
    step0 = [0]*(len(question_tokens)+1) + [2]*(total_len-2-len(question_tokens)) + [0]
    timestep_ids.append(step0)
    for i, sent in enumerate(sentence_tokens):
        this_step = [0]*(len(question_tokens)+1)
        this_step += [1] * sum([len(s) for s in sentence_tokens[:i]])
        this_step += [2] * len(sent) + [3] * sum([len(s) for s in sentence_tokens[i+1:]]) + [0]
        assert len(this_step) == total_len
        timestep_ids.append(this_step)
    return timestep_ids


def compute_state_change_seq(gold_loc_seq):
    num_states = len(gold_loc_seq)
    # whether the entity has been created. (if exists from the beginning, then it should be True)
    create = False if gold_loc_seq[0] == '-' else True
    gold_state_seq = []

    for i in range(1, num_states):

        if gold_loc_seq[i] == '-':  # could be O_C, O_D or D
            if create == True and gold_loc_seq[i-1] == '-':
                gold_state_seq.append('O_D')
            elif create == True and gold_loc_seq[i-1] != '-':
                gold_state_seq.append('D')
            else:
                gold_state_seq.append('O_C')

        elif gold_loc_seq[i] == gold_loc_seq[i-1]:
            gold_state_seq.append('E')

        else:  # location change, could be C or M
            if gold_loc_seq[i-1] == '-':
                create = True
                gold_state_seq.append('C')
            else:
                gold_state_seq.append('M')
    
    assert len(gold_state_seq) == len(gold_loc_seq) - 1
    
    return gold_state_seq

def read_paragraphs():
    prompts = {}
    with open('../data/Paragraphs.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            prompts[row[2]] = row[1]
    return prompts

def reform_prompt(prompt):
    if prompt.startswith('Describe'):
        prompt = prompt.replace('Describe', 'in')
    elif prompt.startswith('How does'):
        prompt = prompt.replace('How does', 'during how')
    elif prompt.startswith('How do'):
        prompt = prompt.replace('How do', 'during how')
    elif prompt.startswith('What happens'):
        prompt = prompt.replace('What happens', '').strip()
    elif prompt.startswith('How are'):
        prompt = prompt.replace('How are', 'during how')
    elif prompt.startswith('How is'):
        prompt = prompt.replace('How is', 'during how')
    elif prompt.startswith('What are'):
        prompt = prompt.replace('What are', 'in')
    elif prompt.startswith('What do'):
        prompt = prompt.replace('What do', 'in')
    elif prompt.startswith('What cause'):
        prompt = prompt.replace('What cause', 'in')
    elif prompt.startswith('What causs'):
        prompt = prompt.replace('What causs', 'in')
    else:
        print ('should not happen')
    if prompt[-1] in string.punctuation:
        prompt = prompt[:-1]
    return prompt

class ProParaDataset(torch.utils.data.Dataset):

    def __init__(self, data, tokenizer, device, add_prompt):
        super().__init__()

        if type(data) == str:
            print (data)
            data = read_propara_data(data)    
        self.tokenizer = tokenizer
        self.add_prompt = add_prompt
        self.device = device
        self.num_spans = Counter()
        self.span_len = Counter()
        self.build_data(data)

    def get_location_spans(self, para, para_bpe, loc):
        spans = []
        if loc not in ['-', '?', 'nil']:
            new_loc = None
            loc_ = nlp(loc)
            para_ = nlp(para)
            loc_tokens = [w.text for w in loc_]
            para_tokens = [w.text for w in para_]
            if not check_loc(loc_tokens, para_tokens):
                loc_lemma = ' '.join([w.lemma_ for w in loc_])
                loc_lemma_tokens = [w.lemma_ for w in loc_]
                if not check_loc(loc_lemma_tokens, para):
                    para_lemma = ' '.join([w.lemma_ for w in para_])
                    para_lemma_tokens = [w.lemma_ for w in para_]
                    if not check_loc(loc_lemma_tokens, para_lemma_tokens):
                        if not check_loc(loc_tokens, para_lemma_tokens):
                            if loc in swap_dict:
                                new_loc = swap_dict[loc]
                            else:
                                span = [-1, -1]
                                spans.append(span)
                                print ('Location not found', loc, loc_lemma)
                                return [s[0] for s in spans], [s[1] for s in spans]
                        else:
                            new_loc = None
                            for i in range(len(para_lemma_tokens)):
                                if para_lemma_tokens[i:i+len(loc_tokens)] == loc_tokens:
                                    new_loc = ' '.join(para_tokens[i:i+len(loc_tokens)])
                                    break
                    else:
                        new_loc = None
                        for i in range(len(para_lemma_tokens)):
                            if para_lemma_tokens[i:i+len(loc_lemma_tokens)] == loc_lemma_tokens:
                                new_loc = ' '.join(para_tokens[i:i+len(loc_lemma_tokens)])
                                break
                else:
                    new_loc = loc_lemma
            else:
                new_loc = loc
            if ',' in new_loc:
                new_loc = new_loc.replace(' ,', ',')
            if '\'s' in new_loc:
                new_loc = new_loc.replace(' \'s', '\'s')
            if '.' in new_loc:
                new_loc = new_loc.replace('.', '').strip()
            if new_loc == 'recycling bin' and 'recycling bins' in para:       # this happens when we migrate to R2D2
                new_loc = 'recycling bins'
            loc_bpe = self.tokenizer.tokenize(new_loc, add_prefix_space=True)
            for i in range(len(para_bpe)):
                if para_bpe[i:i+len(loc_bpe)] == loc_bpe:
                    spans.append([i+1, i+len(loc_bpe)])   # here we bump up 1 to account for [CLS]
            if len(spans) == 0:
                print (loc, new_loc)
                print (para)
                print (new_loc, loc_bpe)
                print (para_bpe)
                print (para_lemma)
                print (loc_tokens)
                print (para_tokens)
                loc_bpe = self.tokenizer.tokenize(new_loc)
                for i in range(len(para_bpe)):
                    if para_bpe[i:i+len(loc_bpe)] == loc_bpe:
                        spans.append([i+1, i+len(loc_bpe)])   # here we bump up 1 to account for [CLS]
                assert len(spans) != 0
            self.span_len[len(loc_bpe)] += 1
        else:
            span = [-1, -1]
            spans.append(span)
        self.num_spans[len(spans)] += 1

        return [s[0] for s in spans], [s[1] for s in spans]

    def build_data(self, data):
        paragraphs = read_paragraphs()
        state2idx = {'O_C': 0, 'O_D': 1, 'E': 2, 'M': 3, 'C': 4, 'D': 5}
        self.dataset = []
        for sample in tqdm(data):
            story = sample['sentence_texts']
            participants = sample['participants']
            prompt = paragraphs[sample['para_id']]
            prompt = reform_prompt(prompt)
            for entity_id, states in enumerate(sample['states']):
                if self.add_prompt:
                    question = "Where is " + str(participants[entity_id]) + f" {prompt} ?</s>"
                else:
                    question = "Where is " + str(participants[entity_id]) + "?</s>"
                para = question + ' ' + ' '.join(story)
                para = para.lower()
                question_tokens = self.tokenizer.tokenize(question.lower())
                sentence_tokens = [self.tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story]
                para_tokens = [w for w in question_tokens]
                for sent in sentence_tokens:
                    para_tokens += sent
                para_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(para_tokens) + [self.tokenizer.sep_token_id]                
                timestep_ids = make_timestep_sequence(question_tokens, sentence_tokens)

                start_positions = []
                end_positions = []
                for step, state in enumerate(states):
                    
                    start_position, end_position = self.get_location_spans(para, para_tokens, state)
                    start_position = [sp if sp > 0 else 0 for sp in start_position]
                    end_position = [ep if ep > 0 else 0 for ep in end_position]

                    start_positions.append(start_position)
                    end_positions.append(end_position)
                state_change_labels = compute_state_change_seq(states)
                state_change_labels = [state2idx[s] for s in state_change_labels]
                exps = {'input_ids': para_ids, 'timestep_type_ids': timestep_ids, 'start_positions': start_positions, 'end_positions': end_positions, 'state_change_labels': state_change_labels}
                self.dataset.append(exps)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        instance = self.dataset[index]
        return instance

    def collate(self, batch):

        batch = batch[0]
        max_num_span = max([len(x) for x in batch['start_positions']])
        input_ids = torch.tensor([batch['input_ids'] for _ in range(len(batch['start_positions']))], device=self.device)
        att_mask = torch.ones_like(input_ids, device=self.device)
        timestep = torch.tensor(batch['timestep_type_ids'], device=self.device)

        start_pos = torch.full((len(batch['start_positions']), max_num_span), -100, device=self.device)
        end_pos = torch.full((len(batch['start_positions']), max_num_span), -100, device=self.device)
        for i in range(len(batch['start_positions'])):
            start_pos[i][:len(batch['start_positions'][i])] = torch.tensor(batch['start_positions'][i])
            end_pos[i][:len(batch['end_positions'][i])] = torch.tensor(batch['end_positions'][i])
        state_change_labels = torch.tensor(batch['state_change_labels'], device=self.device)
        
        return {'input_ids': input_ids,
                'attention_mask': att_mask, 
                'timestep_type_ids': timestep, 'start_positions': start_pos, 'end_positions': end_pos,
                'state_change_labels': state_change_labels, 'num_steps':len(input_ids)}

class MixDataset(ProParaDataset):

    def __init__(self, data, tokenizer, device, add_prompt):
        super().__init__('../data/grids.v1.train.json', tokenizer, device, add_prompt)
        silver_data = read_propara_data(data)
        self.build_data1(silver_data)
        print (len(self.dataset))

    def build_data1(self, data):
        state2idx = {'O_C': 0, 'O_D': 1, 'E': 2, 'M': 3, 'C': 4, 'D': 5}
        for sample in tqdm(data):
            story = sample['sentence_texts']
            participants = sample['participants']
            prompt = sample['prompt']
            prompt = reform_prompt(prompt)
            for entity_id, states in enumerate(sample['states']):
                if self.add_prompt:
                    question = "Where is " + str(participants[entity_id]) + f" {prompt} ?</s>"
                else:
                    question = "Where is " + str(participants[entity_id]) + "?</s>"
                para = question + ' ' + ' '.join(story)
                para = para.lower()
                question_tokens = self.tokenizer.tokenize(question.lower())
                sentence_tokens = [self.tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story]
                para_tokens = [w for w  in question_tokens]
                for sent in sentence_tokens:
                    para_tokens += sent
                para_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(para_tokens) + [self.tokenizer.sep_token_id]                
                timestep_ids = make_timestep_sequence(question_tokens, sentence_tokens)
                
                start_positions = []
                end_positions = []
                for step, state in enumerate(states):
                    state = state.strip()
                    if '.' in state:
                        state = state.split('.')[0].strip()
                    
                    start_position, end_position = self.get_location_spans(para, para_tokens, state)
                    start_position = [sp if sp > 0 else 0 for sp in start_position]
                    end_position = [ep if ep > 0 else 0 for ep in end_position]
                    
                    start_positions.append(start_position)
                    end_positions.append(end_position)
                state_change_labels = compute_state_change_seq(states)
                state_change_labels = [state2idx[s] for s in state_change_labels]
                exps = {'input_ids': para_ids, 'timestep_type_ids': timestep_ids, 'start_positions': start_positions, 'end_positions': end_positions, 'state_change_labels': state_change_labels}
                self.dataset.append(exps)

class TripDataset(torch.utils.data.Dataset):

    def __init__(self, data, tokenizer, device):
        super(TripDataset, self).__init__()

        if type(data) == str:
            print (data)
            data = read_propara_data(data)    
        self.tokenizer = tokenizer
        self.device = device
        self.build_data(data)

    def build_data(self, data):
        self.dataset = []
        plausible_count = Counter()
        conflict_count = Counter()
        for sample in tqdm(data):
            story = sample['sentence_texts']
            participants = sample['participants']
            for entity_id, states in enumerate(sample['states']):
                question = participants[entity_id] + "?!</s>"
                question_tokens = self.tokenizer.tokenize(question.lower())
                sentence_tokens = [self.tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story]
                para_tokens = [w for w  in question_tokens]
                for sent in sentence_tokens:
                    para_tokens += sent
                para_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(para_tokens) + [self.tokenizer.sep_token_id]

                timestep_ids = make_timestep_sequence(question_tokens, sentence_tokens)[1:]
                effect_labels = []
                precondition_labels = []
                conflict_label = 0
                plausible_label = 1
                conflict_span = sample['confl_pairs']

                if len(conflict_span) > 0:
                    conflict_span = conflict_span[-1]
                    conflict_label = 1
                    plausible_label = 0
                for step, state in enumerate(states):
                    effect_labels.append(state[1])
                    precondition_labels.append(state[0])

                real_conflicts = []
                if conflict_label == 1:
                    for i1 in range(len(effect_labels)):
                        for i2 in range(i1+1, len(effect_labels)):
                            if [i1, i2] == conflict_span:
                                real_conflicts.append(1)
                            else:
                                real_conflicts.append(0)
                    
                    conflict_label = real_conflicts.index(1)
                else:
                    conflict_label = -100
                plausible_count[plausible_label] += 1
                conflict_count[conflict_label] += 1
                exp = {'input_ids': [para_ids]*len(timestep_ids), 'attention_mask': [1]*len(para_ids), 'timestep_type_ids': timestep_ids}

                exp['effect_labels'] = effect_labels
                exp['precondition_labels'] = precondition_labels
                exp['conflict_label'] = conflict_label
                exp['plausible_label'] = plausible_label
                self.dataset.append(exp)
        print (len(self.dataset))
        print (plausible_count)
        print (conflict_count)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):

        instance = self.dataset[index]
        return instance

    def collate(self, batch):

        # pad according to max_len
        batch = batch[0]
        input_ids = torch.tensor(batch['input_ids'], device=self.device)
        att_mask = torch.tensor([batch['attention_mask'] for _ in range(len(batch['input_ids']))], device=self.device)
        timestep = torch.tensor(batch['timestep_type_ids'], device=self.device)
        effect_labels = torch.tensor(batch['effect_labels'], device=self.device)
        precondition_labels = torch.tensor(batch['precondition_labels'], device=self.device)
        conflict_label = torch.tensor(batch['conflict_label'], device=self.device)
        plausible_label = torch.tensor(batch['plausible_label'], device=self.device)
        
        return {'input_ids': input_ids,
                'attention_mask': att_mask, 
                'timestep_type_ids': timestep, 'effect_labels': effect_labels, 'precondition_labels': precondition_labels,
                 'conflict_label': conflict_label, 'plausible_label': plausible_label}
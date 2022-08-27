import json

from data_utils import read_propara_data
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score

att_default_values = [('h_location', 0), ('conscious', 2), 
                      ('wearing', 0), ('h_wet', 0), 
                      ('hygiene', 0), ('location', 0), 
                      ('exist', 2), ('clean', 0), 
                      ('power', 0), ('functional', 2), 
                      ('pieces', 0), ('wet', 0), 
                      ('open', 0), ('temperature', 0), 
                      ('solid', 0), ('contain', 0), 
                      ('running', 0), ('moveable', 2), 
                      ('mixed', 0), ('edible', 0)]

def find_conflict_indices(length):
    all_pairs = []
    for i in range(length):
        for j in range(i+1, length):
            all_pairs.append([i, j])
    return all_pairs

def convert_trip_output_format_complete(filename, filename1, all_stories=True):
    pred_outputs = json.load(open(filename, 'r'))
    original_data = read_propara_data(filename1)
    converted_outputs = []
    effects_pred = []
    preconditions_pred = []
    
    for i in range(0, len(pred_outputs), 2):
        sample_pred = {}
        plausible1 = np.mean(np.array([s[3] for s in pred_outputs[i]]), axis=0)
        plausible2 = np.mean(np.array([s[3] for s in pred_outputs[i+1]]), axis=0)
        conf1_counter = np.mean(np.array([s[2] for s in pred_outputs[i]]), axis=0)
        conf2_counter = np.mean(np.array([s[2] for s in pred_outputs[i+1]]), axis=0)
        conflict1 = np.argmax(conf1_counter)
        conflict2 = np.argmax(conf2_counter)

        all_pairs = find_conflict_indices(len(original_data[i]['sentence_texts']))
        if original_data[i]['plausible']:
            sample_pred['story_label'] = 0
            sample_pred['conflict_label'] = original_data[i+1]['confl_pairs'][-1]
            assert len(sample_pred['conflict_label']) == 2
            sample_pred['preconditions_label'] = [[ent[step][0] if step < len(ent) else [0]*20 for step in range(len(original_data[i+1]['sentence_texts']))] for ent in original_data[i+1]['states']]
            sample_pred['effects_label'] = [[ent[step][1] if step < len(ent) else [0]*20 for step in range(len(original_data[i+1]['sentence_texts']))] for ent in original_data[i+1]['states']]
        else:
            sample_pred['story_label'] = 1
            sample_pred['conflict_label'] = original_data[i]['confl_pairs'][-1]
            assert len(sample_pred['conflict_label']) == 2
            sample_pred['preconditions_label'] = [[ent[step][0] if step < len(ent) else [0]*20 for step in range(len(original_data[i]['sentence_texts']))] for ent in original_data[i]['states']]
            sample_pred['effects_label'] = [[ent[step][1] if step < len(ent) else [0]*20 for step in range(len(original_data[i]['sentence_texts']))] for ent in original_data[i]['states']]
        
        if plausible1[0] > plausible2[0]:
            sample_pred['story_pred'] = 1
            sample_pred['conflict_pred'] = all_pairs[conflict1] 
            sample_pred['preconditions_pred'] = [[[s[0][j][step] for j in range(20)] for step in range(len(original_data[i]['sentence_texts']))] for s in pred_outputs[i]]
            sample_pred['effects_pred'] = [[[s[1][j][step] for j in range(20)] for step in range(len(original_data[i]['sentence_texts']))] for s in pred_outputs[i]]
        else:
            sample_pred['story_pred'] = 0
            sample_pred['conflict_pred'] = all_pairs[conflict2] 
            sample_pred['preconditions_pred'] = [[[s[0][j][step] for j in range(20)] for step in range(len(original_data[i+1]['sentence_texts']))] for s in pred_outputs[i+1]]
            sample_pred['effects_pred'] = [[[s[1][j][step] for j in range(20)] for step in range(len(original_data[i+1]['sentence_texts']))] for s in pred_outputs[i+1]]


        converted_outputs.append(sample_pred)
        if all_stories or original_data[i]['plausible']:
            for ei in range(len(original_data[i]['participants'])):
                for si in range(len(original_data[i]['sentence_texts'])):
                    effects_pred.append([pred_outputs[i][ei][1][_][si] for _ in range(20)])
                    preconditions_pred.append([pred_outputs[i][ei][0][_][si] for _ in range(20)])
        if all_stories or original_data[i+1]['plausible']:
            for ei in range(len(original_data[i+1]['participants'])):
                for si in range(len(original_data[i+1]['sentence_texts'])):
                    effects_pred.append([pred_outputs[i+1][ei][1][_][si] for _ in range(20)])
                    preconditions_pred.append([pred_outputs[i+1][ei][0][_][si] for _ in range(20)])
    return converted_outputs, effects_pred, preconditions_pred

def official_evaluate_trip(filename, filename1, filename2, original_data_file, all_entities=None):
    if type(filename) == str:
        pred_outputs = json.load(open(filename))
    else:
        pred_outputs = filename
    if type(filename1) == str:
        effects_pred = json.load(open(filename1))
    else:
        effects_pred = filename1
    if type(filename2) == str:
        preconditions_pred = json.load(open(filename2))
    else:
        preconditions_pred = filename2
    effects_label = []
    preconditions_label = []
    if type(original_data_file) == str:
        original_data = read_propara_data(original_data_file)
    else:
        original_data = original_data_file

    total = 0
    correct = 0
    consistent = 0
    verifiable = 0
    for i in range(0, len(original_data), 2):
        pred = pred_outputs[int(i/2)]
        if all_entities:
            ents = all_entities[int(i/2)][0]
            for ent in ents:
                curr_eid = original_data[i]['participants'].index(ent)
                e = original_data[i]['states'][curr_eid]
                while len(e) < len(original_data[i]['sentence_texts']):
                    e.append([[0] * 20, [0] * 20])
                for si, s in enumerate(e):
                    effects_label.append(s[1])
                    preconditions_label.append(s[0])
            ents = all_entities[int(i/2)][1]
            for ent in ents:
                curr_eid = original_data[i+1]['participants'].index(ent)
                e = original_data[i+1]['states'][curr_eid]
                while len(e) < len(original_data[i+1]['sentence_texts']):
                    e.append([[0] * 20, [0] * 20])
                for si, s in enumerate(e):
                    effects_label.append(s[1])
                    preconditions_label.append(s[0])
        else:
            for ei, e in enumerate(original_data[i]['states']):
                while len(e) < len(original_data[i]['sentence_texts']):
                    e.append([[0] * 20, [0] * 20])
                for si, s in enumerate(e):
                    effects_label.append(s[1])
                    preconditions_label.append(s[0])
                
            for ei, e in enumerate(original_data[i+1]['states']):
                while len(e) < len(original_data[i+1]['sentence_texts']):
                    e.append([[0] * 20, [0] * 20])
                for si, s in enumerate(e):
                    effects_label.append(s[1])
                    preconditions_label.append(s[0])

        if pred['story_label'] == pred['story_pred']:
            correct += 1
            if len(pred['conflict_label']) == len(pred['conflict_pred']) == 2:
                if pred['conflict_label'][0] == pred['conflict_pred'][0] and pred['conflict_label'][1] == pred['conflict_pred'][1]:
                    consistent += 1
                    states_verifiable = True
                    found_states = False
                    # Check that effect of first conflict sentence has states which are correct
                    for sl, sp in [(pred['effects_label'], pred['effects_pred'])]: # Check preconditions and effects
                        for sl_e, sp_e in zip(sl, sp): # Check all entities
                            for si in [pred['conflict_label'][0]]: # Check conflicting sentences
                                sl_es = sl_e[si]
                                sp_es = sp_e[si]
                                for j, p in enumerate(sp_es): # Check all attributes where there's a nontrivial prediction
                                    if p != att_default_values[j][1] and p > 0: # NOTE: p > 0 is required to avoid counting any padding predictions.
                                        found_states = True
                                        if p != sl_es[j]:
                                            states_verifiable = False

                    # Check that precondition of second conflict sentence has states which are correct
                    #if 'preconditions_label' in pred:
                    for sl, sp in [(pred['preconditions_label'], pred['preconditions_pred'])]: # Check preconditions and effects
                        for sl_e, sp_e in zip(sl, sp): # Check all entities        
                            for si in [pred['conflict_label'][1]]: # Check conflicting sentences
                                sl_es = sl_e[si]
                                sp_es = sp_e[si]
                                for j, p in enumerate(sp_es): # Check all attributes where there's a nontrivial prediction
                                    if p != att_default_values[j][1] and p > 0: # NOTE: p > 0 is required to avoid counting any padding predictions.
                                        found_states = True
                                        if p != sl_es[j]:
                                            states_verifiable = False

                    if states_verifiable and found_states:
                        verifiable += 1

        total += 1
    print ('accuracy', correct/total, 'consistency', consistent/total, 'verifibility', verifiable/total)
    effects_label = np.array(effects_label)
    effects_pred = np.array(effects_pred)
    all_f1s = []
    for i in range(20):
        f1 = f1_score(effects_label[:, i], effects_pred[:, i], average='macro')
        all_f1s.append(f1)
    print ('effects', np.mean(all_f1s))

    preconditions_label = np.array(preconditions_label)
    preconditions_pred = np.array(preconditions_pred)
    all_f1s_pre = []
    for i in range(20):
        f1 = f1_score(preconditions_label[:, i], preconditions_pred[:, i], average='macro')
        all_f1s_pre.append(f1)
    print ('preconditions', np.mean(all_f1s_pre))
    return correct/total, consistent/total, verifiable/total, np.mean(all_f1s), np.mean(all_f1s_pre)

if __name__ == '__main__':
    converted_results, converted_effects, converted_preconditions = convert_trip_output_format_complete('../data/best_model_outputs/trip_test_outputs.json',
    '../data/Trip/test.json', True)
    official_evaluate_trip(converted_results, converted_effects, converted_preconditions, '../data/Trip/test.json')
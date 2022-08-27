import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from data_utils import *
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
import argparse
import logging
logger = logging.getLogger(__name__)
from evalQA import print_metrics
from models import RobertaProPara
from tqdm import tqdm
sys.path.append('./evaluator')
from evaluator.our_evaluator import offcial_eval
import matplotlib.pyplot as plt
from stemming.porter2 import stem
import csv

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def location_match(p_loc, g_loc):
    if p_loc == g_loc:
        return True

    p_string = ' %s ' % ' '.join([stem(x) for x in p_loc.lower().replace('"','').split()])
    g_string = ' %s ' % ' '.join([stem(x) for x in g_loc.lower().replace('"','').split()])

    if p_string in g_string:
        #print ("%s === %s" % (p_loc, g_loc))
        return True

    return False

def format_output_actions(spans, state_change_answer, entity, para_id):
    idx2state = {0:'O_C', 1:'O_D', 2:'E', 3:'M', 4:'C', 5:'D'}
    state_change_answer = [idx2state[c] for c in state_change_answer]
    data = []
    destroy = 0
    for step in range(1,len(spans)):
        item = {}
        item['para_id'] = para_id
        item['step'] = step
        item['entity'] = entity
        change = state_change_answer[step-1]
        if destroy == 1 and change != 'O_D':
            change = 'O_D'
        if change == 'E':
            item['action'] = 'NONE'
            if step == 1:
                item['before'] = spans[0]
            else:
                item['before'] = data[step-2]['after']
            item['after'] = item['before']
        elif change == 'C':
            item['action'] = 'CREATE'
            item['before'] = '-'
            item['after'] = spans[step]
        elif change == 'D':
            item['action'] = 'DESTROY'
            if step == 1:
                item['before'] = spans[0]
            else:
                item['before'] = data[step-2]['after']
            item['after'] = '-'
            destroy = 1
        elif change == 'M':
            item['action'] = 'MOVE'
            if step == 1:
                item['before'] = spans[0]
            else:
                item['before'] = data[step-2]['after']
            item['after'] = spans[step]
            if item['before'] == item['after']:
                item['action'] = 'NONE'
        elif change == 'O_C':
            item['action'] = 'NONE'
            item['before'] = '-'
            item['after'] = '-'
        else:
            if step == 1:
                print ('this should not happen')
            item['action'] = 'NONE'
            item['before'] = '-'
            item['after'] = '-'       
        data.append(item)
    return data   

def test_model(args, model, test_set, name="test", iteration=0):
    it_test_location_total = {"?": 0, "Location": 0}
    it_test_location_correct = {"?": 0, "Location": 0}
    it_test_entity_total = 0
    it_test_entity_correct = 0
    idx2state = {0:'O_C', 1:'O_D', 2:'E', 3:'M', 4:'C', 5:'D'}

    data_global = []
    paragraphs = read_paragraphs()
    for sample_id in tqdm(range(len(test_set))):
        sample = test_set[sample_id]
        story = sample['sentence_texts']
        participants = sample['participants']
        prompt = paragraphs[sample['para_id']]
        prompt = reform_prompt(prompt)

        for entity_id, states in enumerate(sample['states']):
            if args.add_prompt:
                question = "Where is " + str(participants[entity_id]) + f" {prompt} ?</s>"
            else:
                question = "Where is " + str(participants[entity_id]) + "?</s>"
            question_tokens = roberta_tokenizer.tokenize(question.lower())
            sentence_tokens = [roberta_tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story]
            para_tokens = [w for w  in question_tokens]
            for sent in sentence_tokens:
                para_tokens += sent

            para_ids = [roberta_tokenizer.cls_token_id] + roberta_tokenizer.convert_tokens_to_ids(para_tokens) + [roberta_tokenizer.sep_token_id]
            timestep_ids = make_timestep_sequence(question_tokens, sentence_tokens)
            batch_input = torch.tensor([para_ids]*len(states), device=args.device)
            timestep_ids = torch.tensor(timestep_ids, device=args.device)
            state_change_labels = compute_state_change_seq(states)

            outputs, _, state_changes = model(input_ids=batch_input, timestep_type_ids=timestep_ids, num_steps=len(sentence_tokens)+1)

            outputs1 = outputs['start_logits'] 
            outputs2 = outputs['end_logits'] 
            new_outputs1 = F.softmax(outputs1, dim=-1).unsqueeze(-1)
            new_outputs2 = F.softmax(outputs2, dim=-1).unsqueeze(1)
            joint_logits = torch.bmm(new_outputs1, new_outputs2)
            spans = []
            for step, state in enumerate(states):
                logits = torch.triu(joint_logits[step]).view(-1)
                sorted_pos = torch.argsort(logits, descending=True)      
                for max_pos in sorted_pos:
                    start_pos = max_pos // len(joint_logits[step])
                    end_pos = max_pos % len(joint_logits[step])
                    if end_pos - start_pos > 5:
                       continue
                    break
                if state == '?':
                    it_test_location_total['?'] += 1
                elif state != '-':
                    it_test_location_total['Location'] += 1
                if start_pos == 0 and end_pos == 0:
                    answer = '?'
                    if state == '?':
                        it_test_location_correct['?'] += 1
                else:
                    answer = roberta_tokenizer.decode(para_ids[start_pos : end_pos + 1])
                    if answer == '-':
                        answer = '?'
                    if location_match(answer, state):
                        it_test_location_correct['Location'] += 1
                spans.append(answer)

            state_change_answer, sumed_scores = model.CRFLayer.decode(emissions=state_changes)
            state_change_answer = state_change_answer[0]  
         
            for i in range(len(state_change_labels)):
                it_test_entity_total += 1
                if idx2state[state_change_answer[i]] == state_change_labels[i]:
                    it_test_entity_correct += 1
            sample_output_global = format_output_actions(spans, state_change_answer, participants[entity_id], sample['para_id'])
            data_global.extend(sample_output_global)
        
    print("The test iteration ", str(iteration), " final results are: ")
    print(it_test_location_total, it_test_location_correct)
    location_accuracy = (it_test_location_correct['?']+it_test_location_correct['Location']) / (it_test_location_total['?']+it_test_location_total['Location'])
    print("The location accuracy is: ", location_accuracy)
    print("The entity accuracy is: ", it_test_entity_correct/it_test_entity_total)
    logger.info("The iteration %s location total %s location correct %s location accuracy %s", str(iteration), it_test_location_total, it_test_location_correct, location_accuracy)
    logger.info("entity accuracy %s", it_test_entity_correct/it_test_entity_total)
    
    csv_file_global = f"{args.output_dir}/{name}_{iteration}_output_global.tsv"

    with open(csv_file_global, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for item in data_global:
            item = [item['para_id'], item['step'], item['entity'], item['action'], item['before'], item['after']]
            tsv_output.writerow(item)

    f1_score = offcial_eval(f'../data/answers/{name}/answers.tsv', csv_file_global)
    if name=="test":
        cat_predictions = []
        swap = {'-': 'null', '?': 'unk'}
        for item in data_global:
            item = [item['para_id'], item['step'], item['entity'], item['before'], item['after']]
            item = [e if e not in swap else swap[e] for e in item]
            cat_predictions.append(item)
        cat1, cat2, cat3, macro, micro = print_metrics(cat_predictions)
        print ("The iteration %s cat1 %s cat2 %s cat3 %s macro %s micro %s", str(iteration), cat1, cat2, cat3, macro, micro)
        logger.info("The iteration %s cat1 %s cat2 %s cat3 %s macro %s micro %s", str(iteration), cat1, cat2, cat3, macro, micro)
        sent_metrics = {'cat1': cat1, 'cat2': cat2, 'cat3': cat3, 'macro': macro, 'micro': micro}

    return f1_score

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args, model):
    if args.train_name == 'default':
        logger.info("loading dataset")
        train_set = ProParaDataset('../data/grids.v1.train.json', tokenizer=roberta_tokenizer, device=args.device, add_prompt=args.add_prompt)
    elif args.train_name == 'augment':
        logger.info("loading mix dataset")
        train_set = MixDataset('../data/grids.data.predicted_silver_train.json', tokenizer=roberta_tokenizer, device=args.device, add_prompt=args.add_prompt)
    else:
        print ('training mode not supported')
        exit()
        
    dev_set = read_propara_data('../data/grids.v1.dev.json') 

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    t_total = len(train_set) // args.gradient_accumulation_steps * args.num_train_epochs
    print ('t_total', t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total)

    best_f1 = 0
    ent_losses = []
    location_losses = []
    dev_f1s = []

    global_step = 0
    logging_losses = [0.0]*4

    train_batch = DataLoader(dataset=train_set, batch_size=args.per_gpu_train_batch_size, shuffle=True, collate_fn=train_set.collate)
    for iteration in tqdm(range(args.num_train_epochs)):
        it_total_loss = 0
        it_ent_loss = 0
        it_loc_loss = 0
        model.train()
        for batch in tqdm(train_batch):
            total_loss = 0
            with autocast():
                outputs, losses, state_changes = model(**batch)

            if losses[0] is not None:
                total_loss += losses[0]
                it_loc_loss += losses[0].item()
                logging_losses[0] += losses[0].item()
            if losses[1] is not None:
                total_loss += losses[1]
                it_ent_loss += losses[1].item()
                logging_losses[1] += losses[1].item()

            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps
            total_loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            it_total_loss += total_loss.item()
            global_step += 1

            if global_step % args.logging_steps == 0:
                logger.info("Iteration %s global step %s location loss %s entity loss %s ", iteration, global_step, (logging_losses[0]-logging_losses[2])/args.logging_steps, (logging_losses[1]-logging_losses[3])/args.logging_steps)
                logging_losses[2] = logging_losses[0]
                logging_losses[3] = logging_losses[1]
           
        print("The iteration loss is: ", it_total_loss)
        location_losses.append(it_loc_loss)
        ent_losses.append(it_ent_loss)
        plt.figure()
        plt.plot(location_losses, label="Location Loss")
        plt.plot(ent_losses, label="Status Loss")
        plt.legend()
        plt.savefig(f'{args.output_dir}/train_plot_loss.png')
        plt.close()
            
        model.eval()
        with torch.no_grad():
            f1_dev = test_model(args, model, dev_set, name="dev", iteration=iteration)
        logger.info("The Dev F1 score is %s", f1_dev)
        
        dev_f1s.append(f1_dev)
        plt.figure()
        plt.plot(dev_f1s, label="F1 score")
        plt.legend()
        plt.savefig(f'{args.output_dir}/dev_f1.png')
        plt.close()

        if best_f1 < f1_dev:
            torch.save(model.state_dict(), f"{args.output_dir}/best_model")
            best_f1 = f1_dev

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_name", default='test', type=str, 
                        help="The test file name.")
    parser.add_argument("--train_name", default='default', type=str, 
                        help="The train mode.")

    parser.add_argument("--max_src_len", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_tgt_len", default=512, type=int,
                        help="Optional target sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--add_prompt", action='store_true',
                        help="Whether to add prompt to paragraph.")
    parser.add_argument("--init_prior", action='store_true',
                        help="Whether to initalize CRF transitions with prior probs.")                  
    parser.add_argument("--no_pretrain", action='store_true',
                        help="Ablate pretraining on SQuAD2.")
    parser.add_argument("--num_states", default=6, type=int,
                        help="Number of possible entity states")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")

    parser.add_argument('--seed', type=int, default=15213,
                        help="random seed for initialization")
    args = parser.parse_args()

    new_out_dir = f"{args.output_dir}/{args.train_name}_seed_{args.seed}"
    if args.do_train or (not args.do_train and os.path.exists(new_out_dir)):
        args.output_dir = new_out_dir

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        print ("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    args.n_gpu = 1
    args.device = device = 'cuda:0'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    if args.do_train:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        handler = logging.FileHandler(os.path.join(args.output_dir, 'train.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        os.system("cp main_procedure.py %s" % os.path.join(args.output_dir, 'main_procedure.py'))
        os.system("cp data_utils.py %s" % os.path.join(args.output_dir, 'data_utils.py'))
        os.system("cp models.py %s" % os.path.join(args.output_dir, 'models.py'))
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    
    set_seed(args)

    logger.info("loading global model")

    model = RobertaProPara.from_pretrained('deepset/roberta-large-squad2', return_dict=True, num_states=args.num_states)
    print (count_parameters(model))

    if args.init_prior:
        print ('initializing CRF transitions with prior probs')
        start_prior, end_prior, transition_prior = compute_priors()
        assert len(start_prior) == args.num_states
        model.CRFLayer.start_transitions = torch.nn.Parameter(torch.tensor(start_prior))
        model.CRFLayer.end_transitions = torch.nn.Parameter(torch.tensor(end_prior))
        model.CRFLayer.transitions = torch.nn.Parameter(torch.tensor(transition_prior))
    model.to(device)
    if args.do_train:
        train(args, model)
    if args.do_eval:
        model.load_state_dict(torch.load(f"{args.output_dir}/best_model"))
        model.to(device)
        test_set = read_propara_data(f'../data/grids.v1.{args.test_name}.json')   
        _ = test_model(args, model, test_set, name=args.test_name, iteration=-1)

if __name__ == '__main__':
    main()
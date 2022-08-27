import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from data_utils import read_propara_data, make_timestep_sequence, TripDataset
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
import argparse
import logging
logger = logging.getLogger(__name__)
from models import RobertaTripNoCRF, RobertaTripWithCRF
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import math
import json
from eval_trip import official_evaluate_trip, convert_trip_output_format_complete

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_count(index, data, size, aid):
    start_count = Counter()
    end_count = Counter()
    transition_count = defaultdict(Counter)
    for sample in data:
        for entity_id, states in enumerate(sample['states']):
            start_count[states[0][aid][index]] += 1
            end_count[states[-1][aid][index]] += 1
            for i in range(1, len(states)):
                transition_count[states[i-1][aid][index]][states[i][aid][index]] += 1
    start_prior = np.full(size, -1e9)
    end_prior = np.full(size, -1e9)
    transition_prior = np.full((size, size), -1e9)
    for k, v in start_count.items():
        start_prior[k] = math.log(v/sum(start_count.values()))
    for k, v in end_count.items():
        end_prior[k] = math.log(v/sum(end_count.values()))
    for k, v in transition_count.items():
        for kk, vv in v.items():
            transition_prior[k][kk] = math.log(vv/sum(v.values()))
    return start_prior, end_prior, transition_prior


def compute_trip_priors():
    data = read_propara_data('../data/Trip/train.json')   
    start_priors_pre = []
    end_priors_pre = []
    transition_priors_pre = []
    start_priors_eff = []
    end_priors_eff = []
    transition_priors_eff = []
    for i in range(20):
        if i != 5:
            s, e, t = get_count(i, data, 3, 0)
        else:
            s, e, t = get_count(i, data, 9, 0)
        start_priors_pre.append(s)
        end_priors_pre.append(e)
        transition_priors_pre.append(t)
        if i != 5:
            s, e, t = get_count(i, data, 3, 1)
        else:
            s, e, t = get_count(i, data, 9, 1)
        start_priors_eff.append(s)
        end_priors_eff.append(e)
        transition_priors_eff.append(t)
    return start_priors_pre, end_priors_pre, transition_priors_pre, start_priors_eff, end_priors_eff, transition_priors_eff, 


def test_model_with_CRF(args, model, test_set, name="test", iteration=0):
    all_preds = []
    for sample in tqdm(test_set):
        story = sample['sentence_texts']
        participants = sample['participants']
        sample_pred = []
        for entity_id, states in enumerate(sample['states']):
            question = participants[entity_id] + "?!</s>"
            
            question_tokens = roberta_tokenizer.tokenize(question.lower())
            sentence_tokens = [roberta_tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story]
            para_tokens = [w for w  in question_tokens]
            for sent in sentence_tokens:
                para_tokens += sent
            para_ids = [roberta_tokenizer.cls_token_id] + roberta_tokenizer.convert_tokens_to_ids(para_tokens) + [roberta_tokenizer.sep_token_id]
            timestep_ids = make_timestep_sequence(question_tokens, sentence_tokens)[1:]

            batch_input = torch.tensor([para_ids]*len(timestep_ids), device=args.device)
            timestep_ids = torch.tensor(timestep_ids, device=args.device)
            
            _, outputs = model(input_ids=batch_input, timestep_type_ids=timestep_ids)            

            full_preconditions = []
            full_effects = []
            for i in range(20):
                pre_answer, _ = model.all_CRFs_pre[i].decode(emissions=outputs[0][i])
                full_preconditions.append(pre_answer[0])
                eff_answer, _ = model.all_CRFs_eff[i].decode(emissions=outputs[1][i])
                full_effects.append(eff_answer[0])

            sample_pred.append([full_preconditions, full_effects, F.softmax(outputs[2], dim=1)[0].tolist(), F.softmax(outputs[3], dim=1)[0].tolist()])   
        all_preds.append(sample_pred)
    with open(f"{args.output_dir}/{name}_{iteration}_output.json", 'w') as fout:
        json.dump(all_preds, fout)            
    print(f"The {name} iteration ", str(iteration), " final results are: ")
    
    converted_results, converted_effects, converted_preconditions = convert_trip_output_format_complete(f"{args.output_dir}/{name}_{iteration}_output.json", f"../data/Trip/{name}.json")
    plausible_acc, conflict_acc, verify_acc, effect_f1, precondition_f1 = official_evaluate_trip(converted_results, converted_effects, converted_preconditions, f"../data/Trip/{name}.json")
    print("The verify Acc is: ", verify_acc, 'conflict acc is', conflict_acc, 'plausible acc is', plausible_acc, 'effect F1 is', effect_f1, 'precondition F1 is', precondition_f1)

    return verify_acc, conflict_acc, plausible_acc, effect_f1, precondition_f1           

def test_model(args, model, test_set, name="test", iteration=0):
    all_preds = []
    for sample in tqdm(test_set):
        story = sample['sentence_texts']
        participants = sample['participants']
        sample_pred = []
        for entity_id, states in enumerate(sample['states']):
            question = participants[entity_id] + "?!</s>"
            
            question_tokens = roberta_tokenizer.tokenize(question.lower())
            sentence_tokens = [roberta_tokenizer.tokenize(sent.lower(), add_prefix_space=True) for sent in story]
            para_tokens = [w for w  in question_tokens]
            for sent in sentence_tokens:
                para_tokens += sent
            para_ids = [roberta_tokenizer.cls_token_id] + roberta_tokenizer.convert_tokens_to_ids(para_tokens) + [roberta_tokenizer.sep_token_id]
            timestep_ids = make_timestep_sequence(question_tokens, sentence_tokens)[1:]

            batch_input = torch.tensor([para_ids]*len(timestep_ids), device=args.device)
            timestep_ids = torch.tensor(timestep_ids, device=args.device)
            
            _, outputs = model(input_ids=batch_input, timestep_type_ids=timestep_ids)            

            full_preconditions = []
            full_effects = []
            for i in range(20):
                full_preconditions.append(torch.argmax(outputs[0][i], dim=-1)[0].tolist())
                full_effects.append(torch.argmax(outputs[1][i], dim=-1)[0].tolist())

            sample_pred.append([full_preconditions, full_effects, F.softmax(outputs[2], dim=1)[0].tolist(), F.softmax(outputs[3], dim=1)[0].tolist()])   
        all_preds.append(sample_pred)
    with open(f"{args.output_dir}/{name}_{iteration}_output.json", 'w') as fout:
        json.dump(all_preds, fout)            
    print(f"The {name} iteration ", str(iteration), " final results are: ")
    
    converted_results, converted_effects, converted_preconditions = convert_trip_output_format_complete(f"{args.output_dir}/{name}_{iteration}_output.json", f"../data/Trip/{name}.json")
    plausible_acc, conflict_acc, verify_acc, effect_f1, precondition_f1 = official_evaluate_trip(converted_results, converted_effects, converted_preconditions, f"../data/Trip/{name}.json")
    print("The verify Acc is: ", verify_acc, 'conflict acc is', conflict_acc, 'plausible acc is', plausible_acc, 'effect F1 is', effect_f1, 'precondition F1 is', precondition_f1)

    return verify_acc, conflict_acc, plausible_acc, effect_f1, precondition_f1    

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args, model):
    logger.info("loading dataset")
    train_set = TripDataset('../data/Trip/train.json', tokenizer=roberta_tokenizer, device=args.device)
    dev_set = read_propara_data('../data/Trip/dev.json') 
    test_set = read_propara_data('../data/Trip/test.json')   

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    t_total = len(train_set) // args.gradient_accumulation_steps * args.num_train_epochs
    print ('t_total', t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total)

    best_sum = 0
    change_losses = []
    conflict_losses = []
    plausible_losses = []

    dev_sum = []

    global_step = 0
    logging_losses = [0.0]*6

    train_batch = DataLoader(dataset=train_set, batch_size=args.per_gpu_train_batch_size, shuffle=True, collate_fn=train_set.collate)
    for iteration in tqdm(range(args.num_train_epochs)):
        
        it_total_loss = 0
        it_change_loss = 0
        it_conflict_loss = 0
        it_plausible_loss = 0
        model.train()
        for batch in tqdm(train_batch):
            total_loss = 0
            with autocast():
                losses, outputs = model(**batch)

            if losses[0] is not None:
                total_loss += losses[0]
                it_change_loss += losses[0].item()
                logging_losses[0] += losses[0].item()
            if losses[1] is not None:
                total_loss += losses[1]
                it_conflict_loss += losses[1].item()
                logging_losses[1] += losses[1].item()
            
            if losses[2] is not None:
                total_loss += losses[2]
                it_plausible_loss += losses[2].item()
                logging_losses[2] += losses[2].item()

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
                logger.info("Iteration %s global step %s change loss %s conflict loss %s plausible loss %s ", iteration, global_step, (logging_losses[0]-logging_losses[3])/args.logging_steps, (logging_losses[1]-logging_losses[4])/args.logging_steps, (logging_losses[2]-logging_losses[5])/args.logging_steps)
                logging_losses[3] = logging_losses[0]
                logging_losses[4] = logging_losses[1]
                logging_losses[5] = logging_losses[2]
           
        print("The iteration loss is: ", it_total_loss)
        change_losses.append(it_change_loss)
        conflict_losses.append(it_conflict_loss)
        plausible_losses.append(it_plausible_loss)
        plt.figure()
        plt.plot(change_losses, label="Change Loss")
        plt.plot(conflict_losses, label="Conflict Loss")
        plt.plot(plausible_losses, label="Plausible Loss")
        plt.legend()
        plt.savefig(f'{args.output_dir}/train_plot_loss.png')
        plt.close()
            
        model.eval()
        with torch.no_grad():
            if args.train_name != 'CRF':
                verify_acc, conflict_acc, plausible_acc, effect_f1, precondition_f1 = test_model(args, model, dev_set, name="dev", iteration=iteration)
            else:
                verify_acc, conflict_acc, plausible_acc, effect_f1, precondition_f1 = test_model_with_CRF(args, model, dev_set, name="dev", iteration=iteration)
        logger.info("The Dev verify accuracy is %s", verify_acc)
        logger.info("The Dev conflict accuracy is %s", conflict_acc)
        logger.info("The Dev plausible accuracy is %s", plausible_acc)
        logger.info("The Dev Effect F1 is %s", effect_f1)
        logger.info("The Dev Precondition F1 is %s", precondition_f1)
        accuracy_sum = verify_acc + conflict_acc + plausible_acc
        
        dev_sum.append(accuracy_sum)
        plt.figure()
        plt.plot(dev_sum, label="Acc Sum")
        plt.legend()
        plt.savefig(f'{args.output_dir}/dev_accuracy.png')
        plt.close()
      
        if best_sum < verify_acc:
            torch.save(model.state_dict(), f"{args.output_dir}/best_model")
            best_sum = verify_acc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_name", default='test', type=str, 
                        help="The test file name.")
    parser.add_argument("--train_name", default='no_CRF', type=str, 
                        help="The train mode.")

    parser.add_argument("--max_src_len", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_tgt_len", default=512, type=int,
                        help="Optional target sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--init_prior", action='store_true',
                        help="Whether to initalize CRF transitions with prior probs.")

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
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")

    parser.add_argument('--seed', type=int, default=42,
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
        os.system("cp main_story.py %s" % os.path.join(args.output_dir, 'main_story.py'))
        os.system("cp data_utils.py %s" % os.path.join(args.output_dir, 'data_utils.py'))
        os.system("cp models.py %s" % os.path.join(args.output_dir, 'models.py'))
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    
    set_seed(args)

    logger.info("loading global model")

    if args.train_name != 'CRF':
        model = RobertaTripNoCRF.from_pretrained('roberta-large', return_dict=True)
    else:
        model = RobertaTripWithCRF.from_pretrained('roberta-large', return_dict=True)
    print (count_parameters(model))

    if args.init_prior:
        start_priors_pre, end_priors_pre, transition_priors_pre, start_priors_eff, end_priors_eff, transition_priors_eff = compute_trip_priors()
        for i in range(20):
            model.all_CRFs_pre[i].start_transitions = torch.nn.Parameter(torch.tensor(start_priors_pre[i]))
            model.all_CRFs_pre[i].end_transitions = torch.nn.Parameter(torch.tensor(end_priors_pre[i]))
            model.all_CRFs_pre[i].transitions = torch.nn.Parameter(torch.tensor(transition_priors_pre[i]))
            model.all_CRFs_eff[i].start_transitions = torch.nn.Parameter(torch.tensor(start_priors_eff[i]))
            model.all_CRFs_eff[i].end_transitions = torch.nn.Parameter(torch.tensor(end_priors_eff[i]))
            model.all_CRFs_eff[i].transitions = torch.nn.Parameter(torch.tensor(transition_priors_eff[i]))
    model.to(device)
    if args.do_train:
        train(args, model)
    if args.do_eval:
        model.load_state_dict(torch.load(f"{args.output_dir}/best_model"))
        model.to(device)
        test_set = read_propara_data(f'../data/Trip/{args.test_name}.json')  
        if args.train_name != 'CRF': 
            _, _, _, _, _ = test_model(args, model, test_set, name=args.test_name, iteration=-1)
        else:
            _, _, _, _, _ = test_model_with_CRF(args, model, test_set, name=args.test_name, iteration=-1)

if __name__ == '__main__':
    main()
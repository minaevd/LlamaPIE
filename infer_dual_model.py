from datasets import load_dataset
import datasets
from pathlib import Path
import re
import numpy as np
import os
from peft import LoraConfig, get_peft_model, TaskType

import torch
from model.device import get_device, empty_cache
from transformers import logging as transformers_logging
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from model.CasualTokenClassificationLlama import LlamaForCausalLM_TokenClassifcation

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import accuracy_score
from peft import PeftModel, PeftConfig
import json
import argparse
# os.environ["WANDB_DISABLED"] = "true"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices= ['MIT', 'Sync_claude', 'Sync_perl', 'Sync_soda'], default='Sync_claude')
parser.add_argument('--save-path', type=str, help="path to save result")
parser.add_argument('--data-root', type=str, required=True, help="folder the evaluation dataset was untarred into")

args = parser.parse_args()

dataset_name = args.dataset
output_samples = f"{args.save_path}/{dataset_name}"
os.makedirs(output_samples, exist_ok = True)

if dataset_name == "MIT":
    from mydatasets.Pipeline_dataset import MIT_sample  as  SingleSample
    output_base = os.path.join(args.data_root, 'MIT_final/')
    input_dirs = []
elif dataset_name == "Sync_claude":
    from mydatasets.Pipeline_dataset import Syn_samples  as  SingleSample
    output_base =  os.path.join(args.data_root, 'synthetic/Test/claude')
    input_dirs = []
elif dataset_name == "Sync_perl":
    from mydatasets.Pipeline_dataset import Syn_samples  as  SingleSample
    output_base =  os.path.join(args.data_root, 'perl/Test/claude')
    input_dirs = []
elif dataset_name == "Sync_soda":
    from mydatasets.Pipeline_dataset import Syn_samples  as  SingleSample
    output_base =  os.path.join(args.data_root, 'soda/Test/claude')
    input_dirs = []
else:
    raise ValueError("dataset not supported!") 

old = False
active_factor = 0
classifier_aware = True
generator_aware = True

device = get_device()

# load weight for small model
tokenizer_small = AutoTokenizer.from_pretrained(
    "tuochao/Llama-3.2-1B-Proactive-Small-Peft",
    pad_token="<|eot_id|>"
    )
tokenizer_small.pad_token = tokenizer_small.eos_token
# base checkpoint has no classifier; the adapter loads the trained one, so skip the "newly initialized" warning
transformers_logging.set_verbosity_error()
model_small = LlamaForCausalLM_TokenClassifcation.from_pretrained(
    "tuochao/Llama-3.2-1B-Proactive-Small-Peft",
    device_map=device, 
    torch_dtype=torch.bfloat16, 
    num_labels = 2)
transformers_logging.set_verbosity_warning()

tokenizer_big = AutoTokenizer.from_pretrained(
    "tuochao/Llama-3.1-8B-Proactive-Big-Peft",
    pad_token="<|eot_id|>"
    )

### load weight for big model
tokenizer_big.pad_token = tokenizer_big.eos_token
model_big = AutoModelForCausalLM.from_pretrained(
    "tuochao/Llama-3.1-8B-Proactive-Big-Peft",
    device_map=device, 
    torch_dtype=torch.bfloat16)

terminators = [
    tokenizer_big.eos_token_id,
    tokenizer_big.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer_big.convert_tokens_to_ids("<|end_of_text|>")
]
response_template = "<|start_header_id|>assistant<|end_header_id|>"

Num_turns = 0

FN = 0
FP = 0
TP = 0
TN = 0

soft_FN = 0
soft_FP = 0
soft_TP = 0
soft_TN = 0

med_FN = 0
med_FP = 0
med_TP = 0
med_TN = 0

true_count = 0
model_count = 0
good_silence = 0
bad_silence = 0

resp_lengths = []
ratios = []
gen_ignore_arr = []
num_resps = []
total_num_turns = []
total_num_spoken = []

for i in range(0, 100):
    # print("----------", i)
    print(f"{i+1}/100 - {model_count}", end="\r")
    empty_cache()
    sample = SingleSample(tokenizer_small, tokenizer_big, output_base, split_set="Test", sample_id = i, input_dirs = input_dirs)
    
    if not sample.valid:
        continue

    Num_turns += sample.count_turn()
    L_cache = 0
    save_id = 0
    previous_id = 0

    save_dir = output_samples  + "/{:05d}".format(i)
    # os.makedirs(save_dir, exist_ok = True)
    curr_turn = ""
    curr_response = ""
    # positive means more responses than needed, neg means less than needed
    thisTurn = 0
    lastTurn = 0
    resp_count = 0
    gen_ignore_count = 0
    hard_this_turn = 0
    while True:
        info = sample.streaming_diaglogue()
        if info is None:
            break

        if "Sync" in dataset_name:
            token, token_history, mask, mask_history, label = info 
            curr_label = label[-1]
            if (curr_label == 0 and len(label) > 1 and label[-2] == 1): # just hit a boundary, update flags
                # print(f"-\t-\t-\t-\t-\t-\t-\t-")
                lastTurn = thisTurn
                thisTurn = 0
                hard_this_turn = 0
        else:
            token, token_history, mask, mask_history = info 

        ### check whether the classifier model is history aware
        if classifier_aware:
            token_small = token_history
            mask_small = mask_history
        else:
            token_small = token
            mask_small = mask

        # print("mask", mask[-1])
        if mask_small[-1] == 0:
            L_cache += 1
            continue # skip the non-special token 

        ### start infer on small model
        intput_ids = token_small.unsqueeze(0).to(model_small.device)
        out1 = model_small(input_ids = intput_ids)
        pred = out1.logits[0].cpu()
        pred[..., 1] += active_factor # make it more proactive
        # print(intput_ids)
        pred = torch.argmax(pred, dim=-1)
        
        new_pred = pred[-1]
        new_mask = mask_small[-1]
        assert(new_mask == 1)
        assert(mask_history[-1] == 1)
        assert(mask[-1] == 1)

        ### start infer on large model
        if generator_aware:
            token_big = token_history
            mask_big = mask_history
        else:
            token_big = token
            mask_big = mask
        responded = 0

        if new_pred == 1:
            print("------------- agent is triggered ----------------")
            curr_diag = tokenizer_small.decode(token_big, skip_special_tokens = True)
            # print(curr_diag)
            input_ids2 = sample.get_gen_inputs(curr_diag, old=old)
            input_ids2 = input_ids2.unsqueeze(0).to(model_big.device)
            outputs = model_big.generate(
                input_ids2,
                attention_mask=torch.ones_like(input_ids2),
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id = tokenizer_big.pad_token_id
            )
            response = outputs[0]#[input_ids.shape[-1]:]
            output = tokenizer_big.decode(response, skip_special_tokens=False)
            output = output.split(response_template)
            raw_diag = output[0]
            response = output[1]
            response = response.replace("<|eot_id|>", "")
            response = response.replace("\n", "")

            ### add whisper back if whisper_aware
            if  len(response) >= 1:
                curr_response = response # model did wshiper sth
                print(curr_diag)
                print("***** Agent *****: ", response)
                sample.insert_whisper(" Agent: " + response)
                responded = 1
                resp_lengths.append(len(response.split(' ')))
                # token_lengths.append(len(outputs[0]))
                resp_count += 1
            else:
                gen_ignore_count += 1
                
        L_cache += 1

        if "Sync" in dataset_name:
            if responded == 1:
                model_count += 1
            if curr_label == 1:
                true_count += 1
            if responded == 0 and new_pred == 1 and curr_label == 1:
                good_silence += 1
            if responded == 0 and new_pred == 1 and curr_label == 0:
                bad_silence += 1
            if responded == 1 and curr_label == 1:
                # hard accuracy
                TP += 1

                # med accuracy
                med_TP += 1

                # soft accuracy
                soft_TP += 1
                if lastTurn < 0: # if we were supposed to resp last turn but didnt, move the "supposed to" to this turn
                    lastTurn += 1
                    thisTurn -= 1
            elif responded == 1 and curr_label == 0:
                # hard accuracy
                FP += 1

                # med accuracy
                if hard_this_turn < 0: # if we were supposed to respond earlier this turn but didn't
                    med_FN -= 1
                    med_TN += 1
                    med_TP += 1
                else:
                    med_FP += 1
                hard_this_turn -= 1

                # soft accuracy
                if lastTurn < 0 or thisTurn < 0:
                    soft_FN -= 1
                    soft_TN += 1
                    soft_TP += 1
                    if lastTurn < 0: # if we were supposed to resp last turn, change that turn to a TN instead of FN
                        lastTurn += 1
                    elif thisTurn < 0: # if we were supposed to resp this turn, change this turn to a TN instead of FN
                        thisTurn += 1
                        med_FN -= 1
                        med_TN += 1
                        med_TP += 1
                else: # man it is an FP
                    thisTurn += 1 # resp too many times this turn
                    soft_FP += 1
            elif responded == 0 and curr_label == 1:
                # hard accuracy
                FN += 1

                # med accuracy
                if hard_this_turn > 0: # if we responded too much this turn
                    med_FP -= 1
                    med_TP += 1 
                    med_TN += 1
                else:
                    med_FN += 1
                hard_this_turn -= 1

                # soft accuracy
                if lastTurn <= 0 and thisTurn <= 0:
                    soft_FN += 1
                    thisTurn -= 1
                else:
                    soft_FP -= 1
                    soft_TP += 1
                    soft_TN += 1
                    if lastTurn > 0: # if resp too early, change it to a TP
                        lastTurn -= 1
                    elif thisTurn > 0: # if we resp too early this turn, also change it to a TP
                        thisTurn -= 1
            else:
                TN += 1
                med_TN += 1
                soft_TN += 1

    tokens, tokens_history = sample.snap_dialogue()
    diag = tokenizer_small.decode(tokens_history, skip_special_tokens = True)

    data = {
        "mem": sample.memory_text,
        "diag": diag
    }
    with open(save_dir + '.json', 'w') as fp:
        json.dump(data, fp, indent = 4)
    # print(TP, FP, FN, TN)
    num_turns = sample.count_turn()
    num_spoken = resp_count
    ratio = num_spoken / (num_turns + num_spoken)
    ratios.append(ratio)

    gen_ignore_arr.append(gen_ignore_count)
    num_resps.append(num_spoken + gen_ignore_count)
    total_num_turns.append(num_turns)
    total_num_spoken.append(num_spoken)


# print('TP', 'FP', 'FN', 'TN')
# print(TP, FP, FN, TN)
# print('soft_TP', 'soft_FP', 'soft_FN', 'soft_TN')
# print(soft_TP, soft_FP, soft_FN, soft_TN)
# print(f'true: {true_count} model: {model_count} good silence: {good_silence} bad silence: {bad_silence}')

def getStats(arr):
    if len(arr) == 0:
        print('nothing happened')
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'count': 0,
            'sum': 0,
            'data': []
        }
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'count': len(arr),
        'sum': sum(arr),
        # 'data': arr
    }

# print(f'response lengths: {getStats(resp_lengths)}')
# print(f'ratios: {getStats(ratios)}')

overall_data = {
    'ratios': getStats(ratios),
    'resp_lengths': getStats(resp_lengths),
    'gen_ignore_counts': getStats(gen_ignore_arr),
    'num_resps': getStats(num_resps),
    'speaker_num_turns': getStats(total_num_turns),
    'num_spoken': getStats(total_num_spoken)

}
if "Sync" in dataset_name:
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    soft_accuracy = (soft_TP + soft_TN) / (soft_TP + soft_TN + soft_FP + soft_FN)
    soft_precision = soft_TP / (soft_TP + soft_FP) 
    soft_recall = soft_TP / (soft_TP + soft_FN)

    med_accuracy = (med_TP + med_TN) / (med_TP + med_TN + med_FP + med_FN)
    med_precision = med_TP / (med_TP + med_FP)
    med_recall = med_TP / (med_TP + med_FN)

    # print(f'accuracy: {accuracy} precision: {precision} recall: {recall}')
    # print(f'soft accuracy: {soft_accuracy} soft precision: {soft_precision} soft recall: {soft_recall}')

    overall_data['hard'] = { 
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    overall_data['med'] = {
        'TP': med_TP,
        'FP': med_FP,
        'FN': med_FN,
        'TN': med_TN,
        'med_accuracy': med_accuracy,
        'med_precision': med_precision,
        'med_recall': med_recall
    }
    overall_data['soft'] = {
        'TP': soft_TP,
        'FP': soft_FP,
        'FN': soft_FN,
        'TN': soft_TN,
        'soft_accuracy': soft_accuracy,
        'soft_precision': soft_precision,
        'soft_recall': soft_recall
    }

with open(f'{output_samples}/data.json', 'w') as f:
    json.dump(overall_data, f, indent = 4)

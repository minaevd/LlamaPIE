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
from mydatasets.Pipeline_dataset import OneSample  as  SingleSample
import argparse
os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', type=str, help="path to load conversation")
parser.add_argument('--save-path', type=str, help="path to save result")

args = parser.parse_args()

output_samples = args.save_path
os.makedirs(output_samples, exist_ok = True)

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


empty_cache()
sample = SingleSample(tokenizer_small, tokenizer_big, conversation_folder=args.input_path)

L_cache = 0

save_dir = output_samples  + "/{:05d}".format(0)
curr_response = ""
thisTurn = 0
lastTurn = 0
resp_count = 0
gen_ignore_count = 0

while True:
    info = sample.streaming_diaglogue()
    if info is None:
        break

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
        input_ids2 = sample.get_gen_inputs(curr_diag)
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
            resp_count += 1
        else:
            gen_ignore_count += 1
            
    L_cache += 1

tokens, tokens_history = sample.snap_dialogue()
diag = tokenizer_small.decode(tokens_history, skip_special_tokens = True)

data = {
    "mem": sample.memory_text,
    "diag": diag
}
with open(save_dir + '.json', 'w') as fp:
    json.dump(data, fp, indent = 4)

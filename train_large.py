from mydatasets.Gen_dataset import Gen_dataset_New
import numpy as np
from datasets import load_dataset
import datasets
from pathlib import Path
import os
from peft import LoraConfig, get_peft_model, TaskType
os.environ["WANDB_DISABLED"] = "true"
import torch
from model.device import get_device, empty_cache
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import accuracy_score
from mydatasets.collator import DataCollatorForCompletionOnlyLM
import argparse
import re


parser = argparse.ArgumentParser()
# Experiment Params
parser.add_argument('--model', type=str,
                    default='llama3-8b',
                    help='name of llm [llama3-8b, llama3_1-8b, llama3_2-1b]')
parser.add_argument('--save_path', type=str,
                    help='Path to save model')

parser.add_argument('--data_root', type=str, required=True,
                    help='Folder the training datasets were untarred into')
args = parser.parse_args()


if args.model == "llama3-8b":
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
elif args.model == "llama3_1-8b":
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
elif args.model == "llama3_2-1b":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
else:
    raise ValueError(f"model {args.model} not supported")

device = get_device()

### loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    pad_token="<|eot_id|>"
    )
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device, 
    torch_dtype=torch.bfloat16)

print(f'model {args.model} loaded!')

dataset = None
dataset_val = None

### change the dataset path based on your file systems
dataset_names = [
    os.path.join(args.data_root, "Final_Generation/Pos_Neg/perl/"),
    os.path.join(args.data_root, "Final_Generation/Pos_Neg/soda/"),
    os.path.join(args.data_root, "Final_Generation/Pos_Neg/synthetic/"),
    os.path.join(args.data_root, "Generation/Pos_Neg/")
]

dataset_prob = [
    1,
    1,
    1,
    0.3
]


# aug_config = {
#     "adapt_to_ASR": 1,
#     "drop_word": 0.75,
#     "swap_silence_speaker": 0.3
# }
aug_config = None
neg_prob = 0.25

dataset = Gen_dataset_New(tokenizer, dataset_names =dataset_names, dataset_probs = dataset_prob, split_set="Train", mem_drop_rate = 0.15, neg_prob = neg_prob, history_aware = True)
dataset_val = Gen_dataset_New(tokenizer, dataset_names=dataset_names, dataset_probs = dataset_prob, split_set="Val", mem_drop_rate = 0, neg_prob = neg_prob, history_aware = True)
print(f"Trainig dataset = {len(dataset)} and val set = {len(dataset_val)}")

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["g_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model.gradient_checkpointing_enable()
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir = args.save_path, 
    eval_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    report_to= "none",
    num_train_epochs = 3,
    save_steps = 400,
    logging_steps = 400,
    batch_eval_metrics = True
)

## define eval metric
# Define compute_metrics
def compute_metrics(pred: EvalPrediction, compute_result: bool):
    predictions, labels = pred.predictions, pred.label_ids
    predictions, labels = predictions.cpu(), labels.cpu()
    predicted_class_indices = np.argmax(predictions, axis=-1)
    flattened_preds = predicted_class_indices.flatten()
    flattened_labels = labels.flatten()
    valid_indices = (flattened_labels != -100)
    flattened_preds = flattened_preds[valid_indices]
    flattened_labels = flattened_labels[valid_indices]
    accuracy = accuracy_score(flattened_labels, flattened_preds)
    return {'accuracy': accuracy}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_val,
    data_collator=DataCollatorForCompletionOnlyLM(instruction_template = None, response_template = "<|start_header_id|>assistant<|end_header_id|>", tokenizer=dataset.tokenizer, mlm=False),
    compute_metrics=compute_metrics,
)

def find_latest_ckpt(root_folder):
    # List all subfolders that match the "checkpoint-xxx" pattern
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")
    folders = []

    # Iterate through all subfolders
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            match = checkpoint_pattern.match(subfolder)
            if match:
                folders.append((subfolder, int(match.group(1))))

    # Find the folder with the largest xxx
    if folders:
        largest_checkpoint = max(folders, key=lambda x: x[1])
        print(f"Folder with the largest ckpt: {largest_checkpoint[0]}")
        return os.path.join(root_folder, largest_checkpoint[0])
    else:
        print("No checkpoint folders found.")
        return None

latest_ckpt = find_latest_ckpt(args.save_path)

if latest_ckpt is None:
    trainer.train(resume_from_checkpoint=False)
else:
    trainer.train(resume_from_checkpoint=latest_ckpt)

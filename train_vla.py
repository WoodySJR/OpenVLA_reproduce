import os, sys, torch
current_dir = "/home/songjunru/VLM"
sys.path.append(current_dir)

import numpy as np
from VLM.utils_vlm import format_data, collate_fn, ActionTokenizer, generate_text_from_sample, generate_actions_from_sample
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import wandb
from trl import SFTTrainer
from functools import partial
import pickle

# system and user prompts
from prompts import user_prompt_prefix_vla, system_prompt_vla
# BitsAndBytesConfig int-4 config
from configs.bitsandbytes_config import bnb_config
# Configure LoRA
from configs.peft_config import peft_config
# Configure training arguments
from configs.training_config import training_args

device = "cuda:0"

# load in the processor (including tokenizer and image processor)
model_id = "Qwen/Qwen2-VL-7B-Instruct"
processor = Qwen2VLProcessor.from_pretrained(model_id, cache_dir="/ssd/songjunru/.cache/models")

# define a collate function for the trainer
collate_fn_for_trainer = partial(
    collate_fn,
    processor=processor,
)

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    cache_dir="/ssd/songjunru/.cache/models"
)

'''
# find all linear layers (suitable for LoRA)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
'''

# initialize wandb for logging (need to input token when logging for the first time)
wandb.init(
    project="qwen2-7b-instruct-trl-sft-RT1",  # change this
    name="test_0305",  # change this
    config=training_args,
)

# add 256 action tokens to the tokenizer (and resize embeddings and lm_head)
for i in range(256):
    processor.tokenizer.add_tokens(f"[action_{i}]")
model.resize_token_embeddings(len(processor.tokenizer))

# load in VLA training data
with open("/ssd/datasets/rt1_training_1.pkl", "rb") as f:
    data_raw = pickle.load(f)

# a small portion of RT-1 Robot Action (43 episodes)
max([sample["episode_index"] for sample in data_raw])

# calculate 1% and 99% percentile of each action dimension
all_actions = np.array([data_raw[i]["action"].tolist() for i in range(len(data_raw))])
action_percentile_1 = np.percentile(all_actions, 1, axis=0)
action_percentile_99 = np.percentile(all_actions, 99, axis=0)

action_tokenizers = []
for i in range(all_actions.shape[1]):
    action_tokenizers.append(ActionTokenizer(bins=256, tokenizer=processor.tokenizer, 
                                             min_action=action_percentile_1[i], max_action=action_percentile_99[i]))

# prepare the training data
samples = []
for i in range(len(data_raw)):
    sample = {}
    sample["image"] = data_raw[i]["image"]
    sample["query"] = user_prompt_prefix_vla + data_raw[i]["instruction"]
    action = ""
    for j in range(all_actions.shape[1]):
        # convert numerical actions to action tokens (eg. [action_0])
        action += action_tokenizers[j].tokenize([data_raw[i]["action"][j]])[0]
    sample["label"] = [action]
    samples.append(sample)

train_dataset = [format_data(sample, system_prompt_vla) for sample in samples]
# use part of the training dataset for evaluation, only for debugging!!
eval_dataset = train_dataset[:1000]

# initialize the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn_for_trainer,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

# train the model
trainer.train()

# save the trained parameters (i.e., the LoRA parameters, embeddings, and lm_head)
trainer.save_model(training_args.output_dir)

# during inference, first load the base model
model_new = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    cache_dir="/ssd/songjunru/.cache/models"
)

# then resize, and load the LoRA parameters
model_new.resize_token_embeddings(len(processor.tokenizer))
model_new.load_adapter(training_args.output_dir)

# generate actions from the model - method 1
output = generate_text_from_sample(model, processor, train_dataset[0], device)
#output = train_dataset[0][2]["content"][0]["text"]
predicted_tokens = processor.tokenizer.encode(output)[-len(action_tokenizers):]
actions_predicted = [action_tokenizers[i].detokenize([predicted_tokens[i]])[0] for i in range(len(predicted_tokens))]

# generate actions from the model - method 2
actions_predicted = generate_actions_from_sample(model, processor, action_tokenizers, train_dataset[0], device)




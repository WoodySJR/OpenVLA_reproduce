import os, sys, torch
current_dir = "/home/songjunru/VLM"
sys.path.append(current_dir)

from datasets import load_dataset
from utils_vlm import format_data, collate_fn, generate_text_from_sample
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import wandb
from trl import SFTTrainer
from functools import partial

# system and user prompts
from prompts import system_prompt_vlm
# BitsAndBytesConfig int-4 config
from configs.bitsandbytes_config import bnb_config
# Configure LoRA
from configs.peft_config import peft_config
# Configure training arguments
from configs.training_config import training_args

device = "cuda:0"

dataset_id = "HuggingFaceM4/ChartQA"
train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, 
                                                         split=['train[:10%]', 'val[:10%]', 'test[:10%]'], 
                                                         cache_dir="/ssd/songjunru/.cache")

# format the data
train_dataset = [format_data(sample, system_prompt_vlm) for sample in train_dataset]
eval_dataset = [format_data(sample, system_prompt_vlm) for sample in eval_dataset]
test_dataset = [format_data(sample, system_prompt_vlm) for sample in test_dataset]

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

# initialize wandb for logging
wandb.init(
    project="qwen2-7b-instruct-trl-sft-ChartQA",  # change this
    name="test_0305",  # change this
    config=training_args,
)

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

# save the trained parameters (only the LoRA parameters)
trainer.save_model(training_args.output_dir)

# during inference, first load the base model
model_new = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    cache_dir="/ssd/songjunru/.cache/models"
)

# then load the LoRA parameters
model_new.load_adapter(training_args.output_dir)

# generate text from the model
generate_text_from_sample(model_new, processor, test_dataset[0], device=device)

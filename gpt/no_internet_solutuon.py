import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from evaluate import load
bleu  = load("bleu")     # default = BLEU-4
rouge = load("rouge")    
AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model_init   = lambda: AutoModelForCausalLM.from_pretrained("gpt2")

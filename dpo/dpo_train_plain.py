import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

from datasets import load_dataset
import argparse
from transformers import TrainerCallback
from copy import deepcopy
import os
import heapq
import pandas as pd
import yaml
import random
import numpy as np
import json

with open("/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/seed.json", "r") as file:
        global_seed = json.load(file)["seed"]
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TopKCheckpointSaver(TrainerCallback):
    def __init__(self, k, save_dir, tokenizer):
        self.k = k
        self.save_dir = save_dir
        self.heap = []  # min-heap of (score, checkpoint_path)
        self.best_path = None
        self.best_score = float('-inf')
        self.tokenizer = tokenizer
        os.makedirs(save_dir, exist_ok=True)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        
        score = metrics["eval_rewards/accuracies"]  # negative for max-heap behavior (lower loss = better)
        ckpt_name = f"checkpoint-step{state.global_step}"

        ckpt_path = os.path.join(self.save_dir, ckpt_name)

        if (self.best_score > float('-inf') and abs(score - self.best_score) > 1e-9) or self.best_score == float('-inf'):
             heapq.heappush(self.heap, (score, ckpt_path))


             if score > self.best_score:
               self.best_score = score
               self.best_path =  ckpt_path
       

             # Save checkpoint
             kwargs["model"].save_pretrained(ckpt_path)
             self.tokenizer.save_pretrained(ckpt_path)

       
             if len(self.heap) > self.k:
                # Remove the worst-performing checkpoint
                worst_score, worst_ckpt_path = heapq.heappop(self.heap)
                print(f'Score eliminated: {worst_score}')
                os.system(f"rm -rf {worst_ckpt_path}")

class Prepare:
    def __init__(self, sft_model_path, train_dataset_path, val_dataset_path = None, test_dataset_path = None):
      
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        self.sft_model_path = sft_model_path
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        def preprocess(example):
            chosen_text = f" Target: {example['chosen']}"
            rejected_text = f" Target: {example['rejected']}"
            prompt_text = f"Source: {example['input']}"

           
            return {
                "chosen": chosen_text,
                "rejected":rejected_text,
                "prompt":prompt_text
            }
        
        
        path_dict = {"train":train_dataset_path}
        if val_dataset_path is not None:
            path_dict["val"] = val_dataset_path

        if test_dataset_path is not None:
            path_dict["test"] = test_dataset_path
        

        # Load dataset (expects a JSONL file with "input" and "output" fields)
        dataset = load_dataset("json", data_files= path_dict)

     
       
        # Set padding token to EOS (GPT-2 doesn't have a default pad token)
      

        #self.tokenized_dataset = dataset.map(tokenize_function, batched=True)
        self.dataset = dataset.map(preprocess, remove_columns=['input'], batched=False)
        self.dataset["train"] = self.dataset["train"].shuffle(seed=42)
        #self.dataset = dataset.map(preprocess, batched=False)



def run_training(config_dict, prepare, callbacks):
    model = AutoModelForCausalLM.from_pretrained(prepare.sft_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    dpo_config = DPOConfig(
        output_dir=config_dict["output_dir"],
        beta=config_dict["beta"],
        learning_rate=config_dict["learning_rate"],
        weight_decay = config_dict["weight_decay"],
        gradient_accumulation_steps=1,
        sync_ref_model = False,
        ref_model_mixup_alpha=config_dict["ref_model_mixup_alpha"],
        ref_model_sync_steps=config_dict["ref_model_sync_steps"],
        lr_scheduler_type=config_dict["lr_scheduler_type"],
        per_device_train_batch_size=64,
        max_prompt_length=32,
        max_completion_length = 32,
        evaluation_strategy="epoch",
        save_strategy=config_dict["save_strategy"],
        logging_strategy="no",
        remove_unused_columns=False,
        num_train_epochs=config_dict["epoch"],
        save_total_limit = config_dict['save_total_limit'],
        load_best_model_at_end=config_dict["load_best_model_at_end"],
        metric_for_best_model = config_dict["metric"],
        greater_is_better=True,
        loss_type = config_dict["loss_type"],
        padding_value = prepare.tokenizer.eos_token_id
        
    )

    trainer = DPOTrainer(
    model,
    ref_model=None,
    args=dpo_config,
    train_dataset=prepare.dataset['train'],
    eval_dataset=prepare.dataset['val'],  # Same set since we're overfitting
    callbacks= callbacks,
    processing_class=prepare.tokenizer
   
)

    trainer.train()
    eval_results = trainer.evaluate()

    return eval_results['eval_rewards/accuracies']  

def train_experiment(prepare, conf):
   
    
    conf['output_dir'] = "none"
    conf["save_strategy"] = "no"
    conf["epoch"] = 11
    conf["save_total_limit"] = None
    conf["load_best_model_at_end"] = False
    conf["metric"] = None
    top_k_checkpoint = TopKCheckpointSaver(k=3, save_dir =  f"{conf['model_save_dir']}/full_train_run", tokenizer=prepare.tokenizer)
    callbacks = [top_k_checkpoint]
    run_training(conf, prepare, callbacks)
    print(f'Best score: {top_k_checkpoint.best_score} achieved by: {top_k_checkpoint.best_path}')





#sft_model_path, train_dataset_path, val_dataset_path = None, test_dataset_path = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_dataset_path", type=str, default= "")
    parser.add_argument("--val_dataset_path", type=str, default=None)
    parser.add_argument("--sft_model_path", type=str, default="")
    parser.add_argument("--conf_path", type = str, default = '')
    ##############################################################3
   
    args = parser.parse_args()
    prepare = Prepare(args.sft_model_path, args.train_dataset_path, args.val_dataset_path)
    
    conf = {}
    with open(args.conf_path, 'r') as file:
        conf = yaml.safe_load(file)
    conf["learning_rate"] = float(conf["learning_rate"])
    conf["weight_decay"] = float(conf["weight_decay"])
    conf["beta"] = float(conf["beta"])
    conf["ref_model_mixup_alpha"] = float(conf["ref_model_mixup_alpha"])
    conf["ref_model_sync_steps"] = int(conf["ref_model_sync_steps"])
    train_experiment(prepare, conf)
    
    


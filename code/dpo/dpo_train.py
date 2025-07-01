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
import random
import numpy as np
import json

with open("/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/seed.json", "r") as file:
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
        self.heap = []
        self.best_path = None
        self.best_score = float('-inf')
        self.tokenizer = tokenizer
        os.makedirs(save_dir, exist_ok=True)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        
        score = metrics["eval_rewards/accuracies"]
        ckpt_name = f"checkpoint-step{state.global_step}"

        ckpt_path = os.path.join(self.save_dir, ckpt_name)
        if (self.best_score > float('-inf') and abs(score - self.best_score) > 1e-9) or self.best_score == float('-inf'):
             heapq.heappush(self.heap, (score, ckpt_path))


             if score > self.best_score:
               self.best_score = score
               self.best_path =  ckpt_path


           
             kwargs["model"].save_pretrained(ckpt_path)
             self.tokenizer.save_pretrained(ckpt_path)


             if len(self.heap) > self.k:
              
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
        

       
        dataset = load_dataset("json", data_files= path_dict)

     
       
      
        self.dataset = dataset.map(preprocess, remove_columns=['input'], batched=False)
        self.dataset["train"] = self.dataset["train"].shuffle(seed=42)
       


def run_training(config_dict, prepare, callbacks, loss_type = 'sigmoid'):
    model = AutoModelForCausalLM.from_pretrained(prepare.sft_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    dpo_config = DPOConfig(
        output_dir=config_dict["output_dir"],
        beta=config_dict["beta"],
        learning_rate=config_dict["learning_rate"],
        weight_decay = config_dict["weight_decay"],
        gradient_accumulation_steps=1,
        sync_ref_model = True,
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
	loss_type = loss_type,
        report_to = "none"
        
    )

    trainer = DPOTrainer(
    model,
    ref_model=None,
    args=dpo_config,
    train_dataset=prepare.dataset['train'],
    eval_dataset=prepare.dataset['val'],  
    callbacks= callbacks,
    processing_class=prepare.tokenizer
   
)

    trainer.train()
    eval_results = trainer.evaluate()

    return eval_results['eval_rewards/accuracies']  

def train_experiment(prepare, model_save_dir, loss_type):

    best_config = {
    "learning_rate": 5e-6,
    "weight_decay": 0.0,
    "beta": 0.1,
    "ref_model_mixup_alpha": 1.0,
    "ref_model_sync_steps": 512,
    "lr_scheduler_type": "linear"
    }

    
    param_grid = {
        "learning_rate"               : [5e-7, 1e-6, 5e-6, 1e-5],
        "weight_decay": [0.0, 0.05, 0.1],
        "beta": [0.05, 0.1, 0.15],
        "ref_model_mixup_alpha": [0.2, 0.6, 1],
        "ref_model_sync_steps":[256, 512, 1024],
        "lr_scheduler_type": ["cosine", "linear"]

    }

  
    
    search_order = list(param_grid.keys())
    run = 0
    rows = []
    best_score = float('-inf')

    for param in search_order:
        print(f"\n Optimizing {param}...")
        best_value = best_config[param]

        for value in param_grid[param]:
            current_config = deepcopy(best_config)
            current_config[param] = value

          
            current_config['output_dir'] = f"{model_save_dir}/run{run}"
            current_config["save_strategy"] = "epoch"
            current_config["save_total_limit"] = 1
            current_config["epoch"] = 3
            current_config["load_best_model_at_end"] = True
            current_config["metric"] = 'eval_rewards/accuracies'
            

            score = run_training(current_config, prepare, [], loss_type)
          
            current_config.update({'score': score, 'index': run})
            rows.append(current_config)

           
            if score > best_score:
                best_score = score
                best_value = value
            run += 1

        best_config[param] = best_value
       
        

    pd.DataFrame(rows).sort_values("score", ascending=False).drop(["save_strategy", "save_total_limit","epoch", "output_dir", "load_best_model_at_end"] , axis = 1).to_csv(f"{model_save_dir}/results.csv", index = False)
    print(f"\nFinal optimized config of value {best_score}:")
    print(best_config)
    
    
    best_config['output_dir'] = "none"
    best_config["save_strategy"] = "no"
    best_config["epoch"] = 20
    best_config["save_total_limit"] = None
    best_config["load_best_model_at_end"] = False
    best_config["metric"] = None
    top_k_checkpoint = TopKCheckpointSaver(k=3, save_dir =  f"{model_save_dir}/full_train_run", tokenizer=prepare.tokenizer)
    callbacks = [top_k_checkpoint]
    run_training(best_config, prepare, callbacks, loss_type)
    print(f'Best score: {top_k_checkpoint.best_score} achieved by: {top_k_checkpoint.best_path}')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model_path", type=str, default="")
    parser.add_argument("--train_dataset_path", type=str, default= "")
    parser.add_argument("--val_dataset_path", type=str, default=None)
    parser.add_argument("--test_dataset_path", type=str, default=None)
    parser.add_argument("--comat_config", type=str, default="./diff_prompter/configs/comat_config.yml")
    parser.add_argument("--gsam_config", type=str, default="./diff_prompter/configs/gsam_config.yml")
    parser.add_argument("--reward_config", type=str, default="./diff_prompter/configs/reward_config.yml")
    parser.add_argument("--img_folder", type = str, default = None)
    parser.add_argument("--dpo_save_path", type = str, default = '')
    parser.add_argument("--loss_type", type = str, default = 'sigmoid')
   
    args = parser.parse_args()
    prepare = Prepare(args.sft_model_path, args.train_dataset_path, args.val_dataset_path)
    
    
    train_experiment(prepare, args.dpo_save_path, args.loss_type)
    
    

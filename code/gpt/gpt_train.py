import itertools
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, EarlyStoppingCallback, logging
)

from datasets import load_dataset
from evaluate import load 
import argparse
import pandas as pd
import random
import json
import sys

with open("/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/seed.json", "r") as file:
        global_seed = json.load(file)["seed"]

random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EvalMetric:
    def __init__(self):

        self.bleu  = load("bleu")    
        self.rouge = load("rouge")   



    
    def eval(self, pred_ids, label_ids, tokenizer):
        SEP = "Target:"
        
        pred_txts  = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_txts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

      
        
        pred_txts  = [t.split(SEP, 1)[-1].strip() for t in pred_txts]
        label_txts = [t.split(SEP, 1)[-1].strip() for t in label_txts]
        
        print(pred_txts[0], flush = True)
        
     

       
        bleu_score = self.bleu.compute(predictions=pred_txts,
                                       references=label_txts)["bleu"]
      

        
        rouge_L     = self.rouge.compute(predictions=pred_txts,
                                    references=label_txts)["rougeL"]       

        avg_score   = (bleu_score + rouge_L) / 2.0
        
        return avg_score
        

        


def make_compute_metrics(model, tokenizer, metric):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        input_txts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        prefixes = [t.split("Target:", 1)[0] + "Target:" for t in input_txts]  
      
        tokenizer.padding_side = "left"
        inputs = tokenizer(
                       prefixes,
                       truncation=True,
                       padding="max_length",  
                       max_length = 32,
                       return_attention_mask=True,
                       return_tensors = "pt"
                     
                     )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
          output = model.generate(
                   input_ids=inputs["input_ids"],
                   attention_mask=inputs["attention_mask"],
                   max_new_tokens  = 32,
                   do_sample       = False,   
                   num_beams       = 8,
                   eos_token_id    = tokenizer.eos_token_id,
                   num_return_sequences=1, 
                )
        
     
        pred_ids = output.cpu().tolist()
        label_ids = labels.tolist()
        score = metric.eval(pred_ids, label_ids, tokenizer)
        tokenizer.padding_side = "right"
        return {"avg_bleu_rouge": score}

    return compute_metrics

def train(train_dataset_path, val_dataset_path, model_save_dir):
    
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    metric = EvalMetric()
    
    dataset = load_dataset("json", data_files={"train": train_dataset_path, "val":  val_dataset_path})
   
    model_name = "gpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
 
    model_init   = lambda: AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
   
    
    def tokenize_function(example):
       


        texts = [f"Source: {inp} Target: {out}" for inp, out in zip(example["input"], example["output"])]

       
        tok = tokenizer(
            texts,
            truncation=True,
            padding="max_length", 
            max_length = 64,
            return_attention_mask=True
        )
       
        tok["labels"] = tok["input_ids"].copy()

        return tok


    tokenized_dataset = dataset.map(tokenize_function, batched=True)


    
    param_grid = {
        "learning_rate"               : [1e-5, 5e-5, 1e-4],
        "weight_decay": [0.0, 0.005, 0.01],
        "warmup_steps": [0, 500]
    }
    

    all_configs = list(itertools.product(*param_grid.values()))
    best_score = float("-inf")
    best_cfg = None
    rows = []
    for i, cfg_vals in enumerate(all_configs, 1):
        cfg = dict(zip(param_grid.keys(), cfg_vals))

        training_args = TrainingArguments(
            output_dir=f"{model_save_dir}/run_{i}",
            evaluation_strategy     = "epoch",
            save_strategy           = "epoch",
            logging_strategy = "no",
            load_best_model_at_end  = True,
            metric_for_best_model   = "avg_bleu_rouge",
            greater_is_better       = True,
            save_total_limit        = 1,
            num_train_epochs=5,
            optim="adamw_torch",
            per_device_train_batch_size=16,
            **cfg                  
        )

        model = model_init().to(device)

        model.gradient_checkpointing_enable()

        trainer = Trainer(
            model         = model,
            args          = training_args,
            train_dataset =  tokenized_dataset["train"],
            eval_dataset  =  tokenized_dataset["val"],
            compute_metrics = make_compute_metrics(model, tokenizer, metric),
        )

        trainer.train()

        score = trainer.evaluate()['eval_avg_bleu_rouge']
        
        if score > best_score:
            best_score = score
            best_cfg = cfg.copy()
        cfg.update({'score': score, 'index': i})
        rows.append(cfg)
        del model
        torch.cuda.empty_cache()

                
    pd.DataFrame(rows).sort_values("score", ascending=True).to_csv(f"{model_save_dir}/results.csv", index = False)
    print(f"Best config of value {best_score}:")
    print(best_cfg)
    training_args = TrainingArguments(
            output_dir=f"{model_save_dir}/best_run",
            evaluation_strategy     = "epoch",
            save_strategy           = "epoch",
            logging_strategy = "no",
            load_best_model_at_end  = True,
            metric_for_best_model   = "avg_bleu_rouge",
            greater_is_better       = True,
            
            save_total_limit        = 1,
            num_train_epochs=20,
            optim="adamw_torch",
            per_device_train_batch_size=16,
            **best_cfg                  
        )

   
    model = model_init()
    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset =  tokenized_dataset["train"],
        eval_dataset  =  tokenized_dataset["val"],
        compute_metrics = make_compute_metrics(model, tokenizer, metric),
        callbacks     = [EarlyStoppingCallback(
                        early_stopping_patience=3)]
    )

    trainer.train()
    print(f"Final best score is {trainer.evaluate()['eval_avg_bleu_rouge']}")
    trainer.save_model(f"{model_save_dir}/best_model")
    tokenizer.save_pretrained(f"{model_save_dir}/best_model")
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/dataset/train_aligned.json")
    parser.add_argument("--val_dataset_path", type=str, default="/scratch/leuven/370/vsc37045/resource/gpt/dataset/val_aligned.json")
    parser.add_argument("--model_save_dir", type = str, default = "/scratch/leuven/370/vsc37045/resource/gpt/sft_aligned")
    
   
    args = parser.parse_args()

    train(args.train_dataset_path, args.val_dataset_path, args.model_save_dir)

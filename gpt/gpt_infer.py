from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import sys
import torch 
from itertools import islice
from torch.nn.functional import log_softmax
import random
import numpy as np

print('dependencies loaded')
with open("/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/seed.json", "r") as file:
        global_seed = json.load(file)["seed"]

random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class PromptOutput:
    def __init__(self, data):
        self.data = data

    def __hash__(self):
        # Hash based only on 'output' field
        return hash(self.data['output'])

    def __eq__(self, other):
        return isinstance(other, PromptOutput) and self.data['output'] == other.data['output']

    def to_dict(self):
        self.data.pop('score')
        return self.data


def single_inference(model, tokenizer, content, savepath = None):
    
    print(f'length content: {len(content)}')
    for i, element in enumerate(content):

        input_prompt = "Source: " + element['input'] + " Target:"
        inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=inputs["attention_mask"],
            num_beams = 8,
        )
        content[i]['output'] = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().split('target:')[-1].strip()

    if savepath is not None:
        with open(savepath, 'w') as f:
            json.dump(content, f, indent=4)
    else:
        return content




def multi_inference(model, tokenizer, content, keep_ret=9, top_p=0.95, temperature=0.3, savepath=None, keep_prev=False):
    final_content = []

    for i, element in enumerate(content):
        new_content = set()

        input_prompt = "Source: " + element['input'] + " Target:"
        inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=32,
            num_return_sequences=30,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=inputs["attention_mask"],
            return_dict_in_generate=True,
            output_scores=True,
        )
       
      
        sequences = outputs.sequences
        
        seq_logps = []                                           # one value per sequence
        prompt_len = inputs["input_ids"].shape[1]                # length of original prompt
        num_sequences = outputs.sequences.size(0)                # bs * num_return_sequences
        gen_len = len(outputs.scores)

        for i in range(num_sequences):
            logp = 0.0
            for t in range(gen_len):
                token_id = outputs.sequences[i, prompt_len + t]
                logits_t = outputs.scores[t][i]                  # (vocab_size,)
                logp += log_softmax(logits_t, dim=-1)[token_id]
            seq_logps.append(logp.item())
        

        for sequence, score in zip(sequences, seq_logps):
            
            generated_text = tokenizer.decode(sequence, skip_special_tokens=True).lower().split('target:')[-1].strip()
          
            new_element = element.copy()
            new_element['output'] = generated_text
            new_element['score'] = float(score)
            new_element['gt'] = False
            new_content.add(PromptOutput(new_element))

        # Sort and keep top `keep_ret`
        new_content = set(sorted(new_content, key=lambda x: x.data['score'], reverse=True)[:keep_ret])

        if keep_prev:
            element['score'] = 0
            element['gt'] = True
            old_len = len(new_content)
            new_content.add(PromptOutput(element))
            #in case the model generates the ground truth, we want to flag it as ground truth
            if old_len == len(new_content):
                new_content.remove(PromptOutput(element))
                new_content.add(PromptOutput(element))


        final_content.extend([obj.to_dict() for obj in new_content])

    if savepath is not None:
        with open(savepath, 'w') as f:
            json.dump(final_content, f, indent=4)
    else:
        return final_content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--num_return", type = int, default = 1)
    parser.add_argument("--top_p", type = float, default = 0.95)
    parser.add_argument("--temperature", type = float, default = 0.3)
    parser.add_argument("--keep_prev", action = 'store_true')
    parser.add_argument("--inference_type", type = str, choices=["multi", "single"], default = "multi")
   
   
    args = parser.parse_args()
    # Load the tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Usign device: {device}')
    if len(args.model_path) == 0:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    with open(args.filename, "r") as file:
        content = json.load(file)
    
    if args.inference_type == 'multi':
        multi_inference(model, tokenizer, content, args.num_return, args.top_p, args.temperature, args.savepath, args.keep_prev)
    
    else:

        single_inference(model, tokenizer, content, args.savepath)

    print('end')


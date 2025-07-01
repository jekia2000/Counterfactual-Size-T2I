from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import torch 
import numpy as np
import random

with open("/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/seed.json", "r") as file:
        global_seed = json.load(file)["seed"]

random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--savepath", type=str, default=None)
   
   
    args = parser.parse_args()
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.filename, "r") as file:
        content = json.load(file)
    
    prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for i, element in enumerate(content):
        plain_text = element['input']
        input_ids = tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
        eos_id = tokenizer.eos_token_id
        outputs = prompter_model.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=1, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        element['output'] = output_texts[0].replace(plain_text+" Rephrase:", "").strip()

    with open(args.savepath, 'w') as f:
            json.dump(content, f, indent=4)


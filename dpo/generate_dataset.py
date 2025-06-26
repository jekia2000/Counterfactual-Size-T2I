from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import argparse
import torch

def get_ds(path, train_save_path, val_save_path, extend_save_path, batch_size):
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
   
   


   
    prompt = """
    [INST] Rewrite the following prompt: {input}
    Only output the rewritten prompt and nothing else.
    [\INST]
    Answer:
    """

    with open(path, 'r') as file:
        content = json.load(file)
    df = pd.DataFrame(content)
   

    groups = df.groupby(by = ["objects"])
    categories_val = set()

    split_column = {}
    for group in groups:
        is_success = False
        split_column_per_group = []
        skip_rows = False
        for i, row in group[1].iterrows():
            if skip_rows:
                split_column_per_group.append('none')

            elif row['reward'] == 1.5:
                is_success = True
                if row['categories'] not in categories_val:
                    categories_val.add(row['categories'])
                    split_column_per_group.append('val')
                    skip_rows = True


                else:
                    split_column_per_group.append('train')

            else:
                split_column_per_group.append('none')
                    
        if not is_success:
            split_column_per_group[0] = 'extend'
        
        split_column[group[0]] = split_column_per_group

    train_ds = []
    val_ds = []
    extend_ds = []

    for group in groups:
        is_success = False
        split_column_per_group = []
        skip_rows = False
        for (i, row), split in zip(group[1].iterrows(), split_column[group[0]]):
            prompts = [prompt.format(input=row['input']) for _ in range(batch_size)]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                repeated_outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=64
                )
            decoded_outputs = tokenizer.batch_decode(repeated_outputs, skip_special_tokens=True)
            unique_answers = set()
            for output in decoded_outputs:
                index = output.find('Answer:') + 7
                answer = (output[index:]).strip()
                unique_answers.add(answer)
            
            ################
            
            unique_answers = list(unique_answers)
            if len(unique_answers) == 1:
                big_object, small_object = row['objects'].split('-')
                unique_answers.append(f"the {small_object} towers over the {big_object}, emphasizing the size difference")
                
            triplet1 = {'rejected':unique_answers[0], 'chosen':row['output'], 'input':row['input'], 'objects':row['objects'], 'categories':row['categories']}
            triplet2 = {'rejected':unique_answers[1], 'chosen':row['output'], 'input':row['input'], 'objects':row['objects'], 'categories':row['categories']}
          
    
            if split == 'train':    
                train_ds.append(triplet1)
                train_ds.append(triplet2)

            elif split == 'val':
                val_ds.append(triplet1)
                val_ds.append(triplet2)

            elif split == 'extend':
                triplet1.pop('chosen')
                triplet2.pop('chosen')
                extend_ds.append(triplet1)
                extend_ds.append(triplet2)
    
    with open(train_save_path, 'w') as file:
        json.dump(train_ds, file, indent = 4)

    with open(val_save_path, 'w') as file:
        json.dump(val_ds, file, indent = 4)


    with open(extend_save_path, 'w') as file:
        json.dump(extend_ds, file, indent = 4)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/evaluated_prompts.json")
    parser.add_argument("--train_save_path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/train.json")
    parser.add_argument("--val_save_path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/val.json")
    parser.add_argument("--extend_save_path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/extend.json")
    parser.add_argument("--batch_size", type=int, default=6)

   
   
    args = parser.parse_args()
    get_ds(args.path, args.train_save_path, args.val_save_path, args.extend_save_path, args.batch_size)
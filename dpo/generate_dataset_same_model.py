import pandas as pd
import json
import argparse
from collections import defaultdict


def get_ds(path, train_save_path, val_save_path):
    

    with open(path, 'r') as file:
        content = json.load(file)
    df = pd.DataFrame(content)
   

    groups = df.groupby(by = ["objects"])
    categories_val = set()

    ###############################
    rejected_prompts = defaultdict(list)
    for group in groups:
        is_positive_here = False
        for _,row in group[1].iterrows():
            if row['reward'] == 1.5:
                is_positive_here = True

            elif row['reward'] < 0:
                if len(rejected_prompts[group[0]]) < 2:
                    rejected_prompts[group[0]].append(row['output'])

        if not(is_positive_here and len(rejected_prompts[group[0]]) == 2):
            rejected_prompts.pop(group[0])

    
    train_ds = []
    val_ds = []
    for group in groups:
        if group[0] not in rejected_prompts:
            continue
        
    
        skip_rows = False
        for i, row in group[1].iterrows():
            if skip_rows:
                continue

            elif row['reward'] == 1.5:
                triplet1 = {'rejected':rejected_prompts[group[0]][0], 'chosen':row['output'], 'input':row['input'], 'objects':row['objects'], 'categories':row['categories']}
                triplet2 = {'rejected':rejected_prompts[group[0]][1], 'chosen':row['output'], 'input':row['input'], 'objects':row['objects'], 'categories':row['categories']}
                if row['categories'] not in categories_val:
                    categories_val.add(row['categories'])
                    skip_rows = True
                    val_ds.append(triplet1)
                    val_ds.append(triplet2)



                else:
                    train_ds.append(triplet1)
                    train_ds.append(triplet2)

           
                    
       
    print(f'Val dataset length: {len(val_ds)}')
    print(f'Train dataset length: {len(train_ds)}')


    with open(train_save_path, 'w') as file:
        json.dump(train_ds, file, indent = 4)

    with open(val_save_path, 'w') as file:
        json.dump(val_ds, file, indent = 4)




    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/evaluated_prompts.json")
    parser.add_argument("--train_save_path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/train_same_model.json")
    parser.add_argument("--val_save_path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/val_same_model.json")


   
   
    args = parser.parse_args()
    get_ds(args.path, args.train_save_path, args.val_save_path)
import json
import pandas as pd
import argparse
from collections import defaultdict

def analize_reward_generation(path):
    with open(path, 'r') as file:
        content = json.load(file)
    df = pd.DataFrame(content)

    groups = df.groupby(by = ["objects"])

    successful = 0
    gt_successful = 0


    distr_prompt_per_instance = defaultdict(int)
    distr_instance_per_category = defaultdict(int)
    
    for group in groups:
        is_success = False
        is_gt_success = False
        prompt_per_instance = 0
        for i, row in group[1].iterrows():
            distr_instance_per_category[row['categories']] += 0
            if row['reward'] == 1.5 and row['gt']:
                is_success = True
                is_gt_success = True
                prompt_per_instance += 1
               

            elif row['reward'] == 1.5:
                is_success = True
                prompt_per_instance += 1
                

        if is_success:
            successful += 1
            distr_prompt_per_instance[prompt_per_instance] += 1
            distr_instance_per_category[row['categories']] += 1
        
        if is_gt_success:
            gt_successful += 1
    
    total_correct = 0
    for element,value in distr_prompt_per_instance.items():
        total_correct += element * value

  
    print(f'Total number of pairs {len(groups)}\nSuccessfull: {successful/len(groups)*100}%\nGT successful: {gt_successful/len(groups)*100}%')
    print(f'Prompts per successul instances: {distr_prompt_per_instance}')
    print(f'Distribution of categories: {distr_instance_per_category} {len(distr_instance_per_category)}')
    print(f'Total number of correct prompts: {total_correct}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/evaluated_prompts.json")
    
   
    args = parser.parse_args()
    analize_reward_generation(args.path)
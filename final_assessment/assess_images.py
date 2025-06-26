import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
evaluator_dir = os.path.join(script_dir, "..", "evaluator")  # Go up one level and into the correct folder

if evaluator_dir not in sys.path:
    sys.path.append(evaluator_dir)  # Add GroundingDINO to Python path
    
import argparse
import json
from eval import Evaluator
from pathlib import Path
from PIL import Image

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--comat_config", type=str, default="./comat.yml")
   
    parser.add_argument("--gsam_config", type=str, default="./gsam.yml")
    parser.add_argument("--reward_config", type=str, default="reward_config.yml")
    parser.add_argument("--savepath", type = str, default = "")
   
    parser.add_argument("--img_folder", type = str, default = "")
    
   
    args = parser.parse_args()

    evaluator = Evaluator(args.comat_config, args.gsam_config, args.reward_config)

    final_content = []
    with open(args.data, "r") as file:
            content = json.load(file)
    n = len(content)

    for i in range(0, n, args.batch_size):
        imgs, new_content = evaluator.use_comat(content[i:min(i + args.batch_size,n)], args.img_folder)
        #imgs = []

        #for element in content:
        #    imgs.append(Image.open(f"{args.img_folder}/{element['objects']}-1.png"))
   
   
       
    new_content, avg_zo, avg_reward  = evaluator.set_rewards(content, imgs)

    print(f'Accuracy according to the evaluator: {avg_zo}')
        
    new_content = sorted(new_content, key = lambda x: x['reward'], reverse = True)
    
    final_content = new_content
    
    with open(args.savepath, "w") as file:
        json.dump(final_content, file, indent = 4)
    



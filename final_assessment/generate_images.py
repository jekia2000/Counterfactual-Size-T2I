import argparse
import json
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
evaluator_dir = os.path.join(script_dir, "..", "evaluator")  # Go up one level and into the correct folder

if evaluator_dir not in sys.path:
    sys.path.append(evaluator_dir)  # Add GroundingDINO to Python path
from eval import Evaluator

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/scratch/leuven/370/vsc37045/resource/dpo/dataset/extended_prompts.json")
  
    parser.add_argument("--comat_config", type=str, default="./comat.yml")
    parser.add_argument("--gsam_config", type=str, default="./gsam.yml")
    parser.add_argument("--reward_config", type=str, default="reward_config.yml")
    parser.add_argument("--savepath", type = str, default = "/scratch/leuven/370/vsc37045/resource/dpo/dataset/evaluated_prompts.json")
   
    parser.add_argument("--img_folder", type = str, default = None)
    parser.add_argument("--batch_size", type = int, default = 1100)
    args = parser.parse_args()

    evaluator = Evaluator(args.comat_config, args.gsam_config, args.reward_config)
    final_content = []
    with open(args.data, "r") as file:
            content = json.load(file)
    n = len(content)
   
    for i in range(0, n, args.batch_size):
        imgs, new_content = evaluator.use_comat(content[i:min(i + args.batch_size,n)], args.img_folder)
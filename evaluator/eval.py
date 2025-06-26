from infer_v1 import main_comat, load_pipeline
from load import main_gsam, load_config_models
import open_clip

import os
import argparse
import json
from PIL import Image
from rewards import SizeReward
import torch
from collections import defaultdict
from pathlib import Path
import random
import numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"

with open("/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/seed.json", "r") as file:
	global_seed = json.load(file)["seed"]
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

class Evaluator:

    def __init__(self, comat_config, gsam_config, reward_config):
        self.reward_obj = SizeReward(reward_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.comat_config = comat_config
        self.gsam_config = gsam_config
    

    
    def use_comat(self, content, img_folder = None):
        self.diffusion_pipe, self.img_size, batch_size = load_pipeline(self.comat_config)
       
        
        imgs = main_comat(self.diffusion_pipe, list(map(lambda x: x['output'], content)), self.img_size, n_iter = 1, batch_size = batch_size)

        del self.diffusion_pipe
        torch.cuda.empty_cache()
        
        captions = list(map(lambda x: x['objects'], content))
            
        name_dict = defaultdict(int)
        if img_folder is not None:
            os.makedirs(img_folder, exist_ok= True)
            for i, (img, caption) in enumerate(zip(imgs, captions)):
                name_dict[caption] += 1 
                img_path = os.path.join(img_folder, f'{caption}-{name_dict[caption]}.png')
                img.save(img_path)
                content[i]['img_path'] = img_path

        return imgs, content



    def image_grid(self, imgs, rows = 1, cols = 1):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid




    def eval_evaluator(self, content): 
        
    
        clip_model_name = "ViT-H-14"
        pretrained = "laion2b_s32b_b79k"
    

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained, device=self.device)
        self.clip_model.eval()
        
        self.dino, self.sam, self.gsam_args =  load_config_models(self.gsam_config)
       
        captions = []
        imgs = []
        names = []
        for image_path, text_prompt in content.items():
            captions.append(text_prompt)
            imgs.append(Image.open(image_path).convert("RGB"))
            names.append(Path(image_path).with_suffix('').name)

        
        areas, return_objects = main_gsam(self.dino, self.sam, self.gsam_args, (self.clip_model, self.clip_preprocess), imgs, captions, names)
        
        
        cnt = 0
        avg_01 = 0
        avg_reward = 0
        new_content = {}
        for (caption, area, image_path) in zip(return_objects, areas, content):
            small, big = caption
            cnt += 1
            big = big.strip()
            small = small.strip()
            area_big = area[big]
            area_small = area[small]

            reward, zero_one_increment = self.reward_obj.get_reward(area_big, area_small)
            avg_01 += zero_one_increment
            avg_reward += reward
            new_content[image_path] = reward


        return new_content, 1.0*avg_01/cnt, 1.0*avg_reward/cnt
    

    def set_rewards(self, content, imgs):
        clip_model_name = "ViT-H-14"
        pretrained = "laion2b_s32b_b79k"
    

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained, device=self.device)
        self.clip_model.eval()
        
        self.dino, self.sam, self.gsam_args =  load_config_models(self.gsam_config)
        
        captions = list(map(lambda x: x['objects'], content))
            
        
        
        areas, return_objects = main_gsam(self.dino, self.sam, self.gsam_args, (self.clip_model, self.clip_preprocess), imgs, captions)

        del self.clip_model
        del self.clip_preprocess
        del self.dino
        del self.sam
        torch.cuda.empty_cache()
        cnt = 0
        avg_01 = 0
        avg_reward = 0
    
        for (caption, area) in zip(return_objects, areas):
            small, big = caption
            cnt += 1
            big = big.strip()
            small = small.strip()
            area_big = area[big]
            area_small = area[small]

            reward, zero_one_increment = self.reward_obj.get_reward(area_big, area_small)
            avg_01 += zero_one_increment
            avg_reward += reward
            content[cnt - 1]['reward'] = reward


        return content, 1.0*avg_01/cnt, 1.0*avg_reward/cnt
      
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/data_path")
    parser.add_argument("--comat_config", type=str, default="./diff_prompter/configs/comat_config.yml")
    parser.add_argument("--gsam_config", type=str, default="./diff_prompter/configs/gsam_config.yml")
    parser.add_argument("--reward_config", type=str, default="./diff_prompter/configs/reward_config.yml")
    parser.add_argument("--k", type = int, default = 5)
    parser.add_argument("--savepath", type = str, default = None)
    parser.add_argument("--img_folder", type = str, default = None)
    

    args = parser.parse_args()
   
    evaluator = Evaluator(args.comat_config, args.gsam_config, args.reward_config)
    with open(args.data, "r") as file:
            content = json.load(file)
    
    content, avg_zo, avg_reward  = evaluator.eval_evaluator(content)
    content = sorted(content.items(), key = lambda x: (x[1], x[0]), reverse = True)
    
    with open(args.savepath, "w") as file:
        json.dump(content, file, indent = 4)

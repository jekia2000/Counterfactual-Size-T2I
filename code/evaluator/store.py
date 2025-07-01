import os
import sys

import numpy as np
import json
import torch
from torchvision.ops import nms
from PIL import Image
import open_clip


import yaml
from types import SimpleNamespace


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import argparse
from pathlib import Path
import cv2

with open("/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/code/seed.json", "r") as file:
        global_seed = json.load(file)["seed"]
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

script_dir = os.path.dirname(os.path.abspath(__file__)) 
gsa_dir = os.path.join(script_dir, "..", "Grounded-Segment-Anything")  
dino_path = os.path.join(gsa_dir, "GroundingDINO")
sam_path = os.path.join(gsa_dir, "segment_anything")
if dino_path not in sys.path:
    sys.path.append(dino_path)  

if sam_path not in sys.path:
    sys.path.append(sam_path)



from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap
from segment_anything import (
sam_model_registry,
SamPredictor
)
def load_image(image_pil):
   
   

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
   
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def store(embed_store, image, boxes, masks, device,  text, clip_t):

    
    model, transform = clip_t
    
    
    total_mask = np.ones(image.shape[:2], dtype = np.uint8)

    
    
    for index, (box, mask) in enumerate(sorted(zip(boxes, masks), key = lambda x: np.sum(x[1]))):
        
        
        x0, y0, x1, y1 = box
        width = x1 - x0 
        height = y1 - y0
        
      
        if width < 32 or height < 32 or np.sum(mask) < 2048:
            continue


      
        if width > height:
            difference = width - height
            if y0 >= difference:
                y0 = y0 - difference
            else:
                y1 = y1 + difference - y0
                y0 = 0
            height = width

        elif height > width:
            difference = height - width
            if x0 >= difference:
                x0 = x0 - difference
            else:
                x1 = x1 + difference - x0
                x0 = 0
            width = height

        
       
        mask = mask[y0:y1, x0:x1].astype(np.uint8)
       
       
        background = np.ones((height, width, 3), dtype=np.uint8) * np.array([255, 255, 240], dtype=np.uint8).reshape(1, 1, -1)
       
      
        image_crop = image[y0:y1, x0:x1].copy()

      
       
        invert_mask = 1 - mask 
        input_img = np.stack([invert_mask] * 3, axis=-1) * background + np.stack([mask] * 3, axis=-1) * image_crop

        
       
        processed_image = transform(Image.fromarray(input_img)).unsqueeze(0).to(device)
        text_tokens = open_clip.tokenize([text]).to(device)
        
        with torch.no_grad():
            image_embedding = model.encode_image(processed_image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

            if 'text' not in embed_store[text]:
                text_embedding = model.encode_text(text_tokens)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                embed_store[text]['text'] = text_embedding.cpu().numpy()[0]
            
           
            embed_store[text]['images'].append(image_embedding.cpu().numpy()[0])
            
    
    



            
        


def get_grounding_output(model, image, box_threshold, text_threshold, iou_threshold, object, H, W, with_logits=True, device="cpu"):
   
    
    logits = []
    boxes = []
    model = model.to(device)
    image = image.to(device)
    tokenlizer = model.tokenizer
    c_orig = [object]
    captions = []
    for i, caption in enumerate(c_orig):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits.append(outputs["pred_logits"].cpu().sigmoid()[0])  
        boxes.append(outputs["pred_boxes"].cpu()[0])  
        captions += [i]*(boxes[-1].shape[0])
    logits = torch.cat(logits, dim=0)
    boxes = torch.cat(boxes, dim = 0)
    captions = torch.tensor(captions, dtype=torch.int)
    
    

    
    logits_filt = logits
    boxes_filt = boxes
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  
    boxes_filt = boxes_filt[filt_mask] 
    captions = captions[filt_mask]
    
   
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    if boxes_filt.size(0) > 0:
        keep = nms(boxes_filt, logits_filt.max(dim=1)[0], iou_threshold)
        boxes_filt = boxes_filt[keep]
        logits_filt = logits_filt[keep]
        captions = captions[keep]

    pred_phrases = []
    boxes = []

    for logit, box, caption in zip(logits_filt, boxes_filt, captions):
        tokenized = tokenlizer(c_orig[caption.item()])
        box = box.unsqueeze(0)
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if len(pred_phrase) > 0:
            if with_logits:
                pred_phrases.append(f"{pred_phrase}|({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
            boxes.append(box)
    
    
    if len(boxes) == 0:
        return [], []
    return torch.cat(boxes, dim = 0), pred_phrases





def load_config_models(config_path):
    
    args = {}
    with open(config_path, 'r') as file:
        args = yaml.safe_load(file)
    args = SimpleNamespace(**args)


    
    config_file = args.config  
    grounded_checkpoint = args.grounded_checkpoint 
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path
    
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
    
    
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    return model, predictor, args

    
def get_stats(embed_store):
    for category in embed_store:
        text_embedding = embed_store[category]['text']
        similarities = [np.dot(image_embedding, text_embedding) for image_embedding in embed_store[category]['images']]
        embed_store[category]['mean'] = np.mean(similarities)
        embed_store[category]['std'] = np.std(similarities, ddof=1)






if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser("CLIP Image-Text Embedding Collector")
    parser.add_argument("--gsam_config", type=str, default="")
    args = parser.parse_args()


    model, predictor, config_args = load_config_models(args.gsam_config)

    box_threshold = config_args.box_threshold
    text_threshold = config_args.text_threshold
    iou_t = config_args.iou_threshold
    
    em_save_path = config_args.embeddings
    img_path = config_args.img_path
    device = config_args.device
    
    clip_model_name = "ViT-H-14"
    pretrained = "laion2b_s32b_b79k"
    

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained, device=config_args.device)
    clip_model.eval()

    clip_t = (clip_model, clip_preprocess)

    embed_store = {}

    
    for category_path in Path(img_path).iterdir():
       
       
        if category_path.is_dir():
            category = category_path.name
            embed_store[category] = {'images': []}
            for image_path in category_path.iterdir():
               
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image = cv2.imread(str(image_path))
                    try:
                       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except:
                        print(f'INVALID: {image_path}')
                        continue
                    
                    
                    H, W = image.shape[0], image.shape[1]
                   
                    delta = abs(H - W) // 2

                    if H > W:
                        left_pad = delta
                        right_pad = abs(H - W) - left_pad
                        image = cv2.copyMakeBorder(image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(240, 255, 255))
                    elif W > H:
                        bottom = delta
                        top = abs(H - W) - bottom
                        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(240, 255, 255))
                        
                    image = cv2.resize(image, (512, 512))
                    image_pil = Image.fromarray(image)

                    dino_image = load_image(image_pil)
                  
                    boxes_filt, pred_phrases = get_grounding_output(
                        model, dino_image, box_threshold, text_threshold, iou_t, category, 512, 512, device=device
                        )  
                    
                    
                    if len(boxes_filt) == 0:
                        continue
                    

                    predictor.set_image(image)
                    boxes_filt = boxes_filt.cpu()

                    
                   
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

                    masks, _, _ = predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes.to(device),
                        multimask_output = False,
                    )
                    
                    
                    store(embed_store, image, boxes_filt.numpy().astype(np.uint16), masks.to('cpu').squeeze(1).numpy().astype(np.uint8), device,  category, clip_t)
                
    get_stats(embed_store)

 
    flat_store = {}
    for category, data in embed_store.items():
        flat_store[f"{category}_text"] = data['text']
        flat_store[f"{category}_images"] = np.array(data['images'])
        flat_store[f"{category}_mean"] = np.array(data['mean'])
        flat_store[f"{category}_std"] = np.array(data['std'])

    np.savez(em_save_path, **flat_store)


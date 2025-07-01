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
    image, _ = transform(image_pil, None)  # 3, h, w
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

def reassign_labels_1(embed_store, image, boxes, masks, device,  big_object, small_object, phrases,  clip_t, full_picture,  change_boxes = False):

    def jaccard_similarity(term, str1, str2):
        set1, set2, term_set = set(str1.split()),set(str2.split()), set(term.split())
        j1 = len(set1 & term_set) / len(set1 | term_set)
        j2 = len(set2 & term_set) / len(set2 | term_set)
        
        if j1 > j2:
            return str1
        else:
            return str2
   
    areas = {big_object : float('-inf'), small_object : float('-inf')}  

    for index, (box, mask) in enumerate(zip(boxes, masks)):
        
        
        x0, y0, x1, y1 = box
        width = x1 - x0 
        height = y1 - y0

      
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

        object = phrases[index].split('|')[0]

        object = jaccard_similarity(object, small_object, big_object)
        phrases[index] = object
        areas[object] = max(areas[object], 0.0 + np.sum(mask))

    if areas[big_object] == float('-inf'):
        areas[big_object] = float('inf')

    return areas, boxes, masks, phrases

def reassign_labels_2(embed_store, image, boxes, masks, device,  big_object, small_object, phrases,  clip_t, full_picture, change_boxes = False):

    def jaccard_similarity(term, str1, str2):
        set1, set2, term_set = set(str1.split()),set(str2.split()), set(term.split())
        j1 = len(set1 & term_set) / len(set1 | term_set)
        j2 = len(set2 & term_set) / len(set2 | term_set)
        
        if j1 > j2:
            return str1
        else:
            return str2
   
    areas = {big_object : float('-inf'), small_object : float('-inf')}

    total_mask = np.ones(image.shape[:2], dtype = np.uint8)
   
    indices_to_remove = []
    dimension = image.shape[0]
    zip_b_m = sorted(zip(boxes, masks, phrases), key = lambda x: np.sum(x[1]))
    masks = np.array(list(map(lambda x: x[1], zip_b_m)))
    boxes = np.array(list(map(lambda x: x[0], zip_b_m)))
    phrases = list(map(lambda x: x[2], zip_b_m))
    for index, (box, mask, _) in enumerate(zip_b_m):
        
        mask = mask & total_mask
        total_mask = total_mask & (1 - mask)
        
        x0, y0, x1, y1 = box
        width = x1 - x0 
        height = y1 - y0

        if width < 32 or height < 32 or np.sum(mask) < 2048:
            indices_to_remove.append(index)
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

        if change_boxes:
            masks[index] = mask
            bound_x0 = dimension
            bound_y0 = dimension
            bound_x1 = -1
            bound_y1 = -1
            for i in range(y0, y1):
                for j in range(x0, x1):
                    if mask[i][j] == 1:
                         bound_x0 = min(bound_x0, j)
                         bound_y0 = min(bound_y0, i)
                         bound_x1 = max(bound_x1, j)
                         bound_y1 = max(bound_y1, i)
            bound_x0 = max(0, bound_x0 - 5)
            bound_y0 = max(0, bound_y0 - 5)
            bound_x1 = min(dimension - 1, bound_x1 + 5)
            bound_y1 = min(dimension - 1, bound_y1 + 5)
            boxes[index][0] = bound_x0
            boxes[index][1] = bound_y0
            boxes[index][2] = bound_x1
            boxes[index][3] = bound_y1


        mask = mask[y0:y1, x0:x1].astype(np.uint8)

       

        object = phrases[index].split('|')[0]

        object = jaccard_similarity(object, small_object, big_object)
        phrases[index] = object
        areas[object] = max(areas[object], 0.0 + np.sum(mask))

    if areas[big_object] == float('-inf'):
        areas[big_object] = float('inf')

    if change_boxes:
        boxes = np.delete(boxes, indices_to_remove, axis=0)
        masks = np.delete(masks, indices_to_remove, axis=0)
        for idx in sorted(indices_to_remove, reverse=True):
               del phrases[idx]
    
    return areas, boxes, masks, phrases


def reassign_labels_3(embed_store, image, boxes, masks, device,  big_object, small_object, phrases,  clip_t, full_picture,  change_boxes = False):

   
    areas = {big_object : float('-inf'), small_object : float('-inf')}
   
    model, transform = clip_t
    
    
    total_mask = np.ones(image.shape[:2], dtype = np.uint8)

    
    indices_to_remove = []
    dimension = image.shape[0]
    zip_b_m = sorted(zip(boxes, masks, phrases), key = lambda x: np.sum(x[1]))
    masks = np.array(list(map(lambda x: x[1], zip_b_m)))
    boxes = np.array(list(map(lambda x: x[0], zip_b_m)))
    phrases = list(map(lambda x: x[2], zip_b_m))
    for index, (box, mask, _) in enumerate(zip_b_m):
        
        mask = mask & total_mask
        total_mask = total_mask & (1 - mask)
     
        
        x0, y0, x1, y1 = box
        width = x1 - x0 
        height = y1 - y0

      
        if width < 32 or height < 32 or np.sum(mask) < 2048:
            indices_to_remove.append(index)
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

        if change_boxes:
            masks[index] = mask
            bound_x0 = dimension
            bound_y0 = dimension
            bound_x1 = -1
            bound_y1 = -1
            for i in range(y0, y1):
                for j in range(x0, x1):
                    if mask[i][j] == 1:
                         bound_x0 = min(bound_x0, j)
                         bound_y0 = min(bound_y0, i)
                         bound_x1 = max(bound_x1, j)
                         bound_y1 = max(bound_y1, i)
            bound_x0 = max(0, bound_x0 - 5)
            bound_y0 = max(0, bound_y0 - 5)
            bound_x1 = min(dimension - 1, bound_x1 + 5)
            bound_y1 = min(dimension - 1, bound_y1 + 5)
            boxes[index][0] = bound_x0
            boxes[index][1] = bound_y0
            boxes[index][2] = bound_x1
            boxes[index][3] = bound_y1
        mask = mask[y0:y1, x0:x1].astype(np.uint8)
       
        background = np.ones((height, width, 3), dtype=np.uint8) * np.array([255, 255, 240], dtype=np.uint8).reshape(1, 1, -1)
       
      
        image_crop = image[y0:y1, x0:x1].copy()
       
        invert_mask = 1 - mask 
        input_img = np.stack([invert_mask] * 3, axis=-1) * background + np.stack([mask] * 3, axis=-1) * image_crop

        
       
        processed_image = transform(Image.fromarray(input_img)).unsqueeze(0).to(device)
            
        with torch.no_grad():
            embedding = model.encode_image(processed_image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
          

        
        embedding = embedding.cpu().numpy()[0]
    
      
        small_image_embeddings = embed_store[small_object + "_images"]
        big_image_embeddings = embed_store[big_object + "_images"]

        small_similarity = float(np.max(np.array([embedding @ img_embedding for img_embedding in small_image_embeddings])))
        big_similarity = float(np.max(np.array([embedding @ img_embedding for img_embedding in big_image_embeddings])))

        if big_similarity > small_similarity:
            areas[big_object] = max(areas[big_object], 0.0 + np.sum(mask))
            phrases[index] = big_object
        elif small_similarity:
            areas[small_object] = max(areas[small_object], 0.0 + np.sum(mask))
            phrases[index] = small_object

   
    
    if areas[big_object] == float('-inf'):
        areas[big_object] = float('inf')
    
    if change_boxes:
        boxes = np.delete(boxes, indices_to_remove, axis=0)
        masks = np.delete(masks, indices_to_remove, axis=0)
        for idx in sorted(indices_to_remove, reverse=True):
               del phrases[idx]

    return areas, boxes, masks, phrases


def reassign_labels_4(embed_store, image, boxes, masks, device,  big_object, small_object, phrases,  clip_t, full_picture,  change_boxes = False):

   
    areas = {big_object : float('-inf'), small_object : float('-inf')}
   
    model, transform = clip_t
    
    
    total_mask = np.ones(image.shape[:2], dtype = np.uint8)

    
    indices_to_remove = []
    dimension = image.shape[0]
    zip_b_m = sorted(zip(boxes, masks, phrases), key = lambda x: np.sum(x[1]))
    masks = np.array(list(map(lambda x: x[1], zip_b_m)))
    boxes = np.array(list(map(lambda x: x[0], zip_b_m)))
    phrases = list(map(lambda x: x[2], zip_b_m))
    for index, (box, mask, _) in enumerate(zip_b_m):

        
        mask = mask & total_mask
        total_mask = total_mask & (1 - mask)
     
        
        x0, y0, x1, y1 = box
        width = x1 - x0 
        height = y1 - y0

      
        if width < 32 or height < 32 or np.sum(mask) < 2048:
            indices_to_remove.append(index)
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

        if change_boxes:
            masks[index] = mask
            bound_x0 = dimension
            bound_y0 = dimension
            bound_x1 = -1
            bound_y1 = -1
            for i in range(y0, y1):
                for j in range(x0, x1):
                    if mask[i][j] == 1:
                         bound_x0 = min(bound_x0, j)
                         bound_y0 = min(bound_y0, i)
                         bound_x1 = max(bound_x1, j)
                         bound_y1 = max(bound_y1, i)
            bound_x0 = max(0, bound_x0 - 5)
            bound_y0 = max(0, bound_y0 - 5)
            bound_x1 = min(dimension - 1, bound_x1 + 5)
            bound_y1 = min(dimension - 1, bound_y1 + 5)
            boxes[index][0] = bound_x0
            boxes[index][1] = bound_y0
            boxes[index][2] = bound_x1
            boxes[index][3] = bound_y1
        mask = mask[y0:y1, x0:x1].astype(np.uint8)
       
        background = np.ones((height, width, 3), dtype=np.uint8) * np.array([255, 255, 240], dtype=np.uint8).reshape(1, 1, -1)
       
      
        image_crop = image[y0:y1, x0:x1].copy()
       
        invert_mask = 1 - mask  
        input_img = np.stack([invert_mask] * 3, axis=-1) * background + np.stack([mask] * 3, axis=-1) * image_crop

        
       
        processed_image = transform(Image.fromarray(input_img)).unsqueeze(0).to(device)
            
        with torch.no_grad():
            embedding = model.encode_image(processed_image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
          

        
        embedding = embedding.cpu().numpy()[0]
    
      
        small_image_embeddings = embed_store[small_object + "_images"]
        big_image_embeddings = embed_store[big_object + "_images"]

        small_similarity = float(np.max(np.array([embedding @ img_embedding for img_embedding in small_image_embeddings])))
        big_similarity = float(np.max(np.array([embedding @ img_embedding for img_embedding in big_image_embeddings])))

        if big_similarity > small_similarity:
            areas[big_object] = max(areas[big_object], 0.0 + np.sum(mask))
            phrases[index] = big_object
        elif small_similarity:
            areas[small_object] = max(areas[small_object], 0.0 + np.sum(mask))
            phrases[index] = small_object

   
    
    if full_picture < 0.33 and areas[big_object] != float('-inf') and areas[small_object] != float('-inf'):
   
        areas[big_object] = float('inf')
        if change_boxes:
          for j_indx in range(len(phrases)):
              if phrases[j_indx] == big_object and not j_indx in indices_to_remove:
                   indices_to_remove.append(j_indx)
    
    if areas[big_object] == float('-inf'):
        areas[big_object] = float('inf')
    
    if change_boxes:
        boxes = np.delete(boxes, indices_to_remove, axis=0)
        masks = np.delete(masks, indices_to_remove, axis=0)
        for idx in sorted(indices_to_remove, reverse=True):
               del phrases[idx]
    return areas, boxes, masks, phrases



            
        


def get_grounding_output(model, image, caption, box_threshold, text_threshold, iou_threshold, small_object, big_object, H, W, with_logits=True, device="cpu"):
   
    
    logits = []
    boxes = []
    model = model.to(device)
    image = image.to(device)
    tokenlizer = model.tokenizer
    c_orig = [small_object, big_object]
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



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    
    
    ax.add_patch(plt.Rectangle(
        (x0, y0), w, h,
        edgecolor='red',
        facecolor='none',
        lw=5
    ))
    
  
    ax.text(
        x0, y0, label,
        color=(1.0, 0.431, 0.78),  
        fontsize=18,              
        weight='bold',
        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
    )


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

    

def main_gsam(model, predictor, args, clip_t, imgs, prompts, names = None):
    
    
    em_file = args.embeddings
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    iou_t = args.iou_threshold
    class_path = args.class_path
    eval_type = args.eval_type
    img_save_folder = args.img_save_folder
    os.makedirs(img_save_folder, exist_ok= True)
    df_objects = pd.read_csv(class_path)
    big_objects = df_objects[df_objects['type'] == 'big']['names']
    small_objects = df_objects[df_objects['type'] == 'small']['names']
    objects = {}
    for object in big_objects:
        objects[object] = "b"
    for object in small_objects:
        objects[object] = "s"

    total_areas = []
    return_objects = []

    label_func = None
    if eval_type == 1:
        label_func = reassign_labels_1
    elif eval_type == 2:
        label_func = reassign_labels_2

    elif eval_type == 3:
        label_func = reassign_labels_3

    elif eval_type == 4:
        label_func = reassign_labels_4

    else:
        raise Exception('Invalid evaluation type')

    
    for idx, (image_pil, text_prompt) in enumerate(zip(imgs, prompts)):
        
        dino_image = load_image(image_pil)
        
        size = image_pil.size
        H, W = size[1], size[0]

        object1, object2 = text_prompt.split('-')
        small_object = None
        big_object = None
        if objects[object1] == "s" and objects[object2] == "b":
                small_object = object1
                big_object = object2
        elif objects[object1] == "b" and objects[object2] == "s":
            big_object = object1
            small_object = object2
        else:
            raise "Problem with the prompt!"

      
       
        image = np.array(image_pil)
        
        modelc, transform = clip_t
        processed_image = transform(image_pil).unsqueeze(0).to(device)
        text_tokens = open_clip.tokenize([f"{small_object} and {big_object}"]).to(device)
        with torch.no_grad():
            embedding = modelc.encode_image(processed_image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  
            text_embedding = modelc.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        full_picture = float(embedding.cpu().numpy()[0] @ text_embedding.cpu().numpy()[0])


        if eval_type == 4:
            if full_picture >= 0.39:
                boxes_filt, pred_phrases = get_grounding_output(
                model, dino_image, text_prompt, 0.2, 0.2, iou_t, small_object, big_object, H, W, device=device
                )

            else:
                boxes_filt, pred_phrases = get_grounding_output(
                model, dino_image, text_prompt, box_threshold, text_threshold, iou_t, small_object, big_object, H, W, device=device
                )  
        else:
            boxes_filt, pred_phrases = get_grounding_output(
                model, dino_image, text_prompt, box_threshold, text_threshold, iou_t, small_object, big_object, H, W, device=device
                )  
        
                
        

        return_objects.append((small_object, big_object))
        if len(boxes_filt) == 0:
            areas = {big_object : float('inf'), small_object : float('-inf')}
            total_areas.append(areas)
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
        
        embed_store = np.load(em_file)
        areas, boxes, masks, pred_phrases = label_func(embed_store, image, boxes_filt.numpy().astype(np.uint16), masks.to('cpu').squeeze(1).numpy().astype(np.uint8), device,  big_object,small_object, pred_phrases, clip_t, full_picture, img_save_folder!="none")
        total_areas.append(areas)
        
     
        if img_save_folder != "none":
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask, plt.gca(), random_color=True)
            for box, label in zip(boxes, pred_phrases):
                show_box(box, plt.gca(), label)

            plt.axis('off')
            plt.savefig(
                os.path.join(f'{img_save_folder}', names[idx] + '.png'),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
        
    return total_areas, return_objects
          

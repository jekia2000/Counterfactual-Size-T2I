import argparse
import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import cv2
#import clip
import open_clip


def store(embed_store, image, device, text_prompt, clip_t):
    h, w, _ = image.shape
    delta = abs(h - w) // 2

    if h > w:
        left_pad = delta
        right_pad = abs(h - w) - left_pad
        image = cv2.copyMakeBorder(image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(240, 255, 255))
    elif w > h:
        bottom = delta
        top = abs(h - w) - bottom
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(240, 255, 255))

    model, transform = clip_t

    processed_image = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    text_tokens = open_clip.tokenize([text_prompt]).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(processed_image)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

        if 'text' not in embed_store[text_prompt]:
            text_embedding = model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            embed_store[text_prompt]['text'] = text_embedding.cpu().numpy()[0]

    embed_store[text_prompt]['images'].append(image_embedding.cpu().numpy()[0])


def get_stats(embed_store):
    for category in embed_store:
        text_embedding = embed_store[category]['text']
        similarities = [np.dot(image_embedding, text_embedding) for image_embedding in embed_store[category]['images']]
        embed_store[category]['mean'] = np.mean(similarities)
        embed_store[category]['std'] = np.std(similarities, ddof=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CLIP Image-Text Embedding Collector")
    parser.add_argument("--embeddings", "-e", type=str, required=True, help="directory to save embeddings")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--img_path", type=str, required=True, help="path to image folder with category subfolders")
    args = parser.parse_args()

    device = args.device
    em_dir = args.embeddings
    img_path = args.img_path

    os.makedirs(em_dir, exist_ok=True)

    # Load open_clip model
    model_name = "ViT-H-14"
    pretrained = "laion2b_s32b_b79k"
    

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()

    clip_t = (model, preprocess)

    embed_store = {}

    for category_path in Path(img_path).iterdir():
        if category_path.is_dir():
            category = category_path.name
            embed_store[category] = {'images': []}
            for image_path in category_path.iterdir():
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image = cv2.imread(str(image_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    store(embed_store, image, device, category, clip_t)

    get_stats(embed_store)

    # Convert nested structure to flat dict for saving
    flat_store = {}
    for category, data in embed_store.items():
        flat_store[f"{category}_text"] = data['text']
        flat_store[f"{category}_images"] = np.array(data['images'])
        flat_store[f"{category}_mean"] = np.array(data['mean'])
        flat_store[f"{category}_std"] = np.array(data['std'])

    np.savez(f"{em_dir}/stored_embeddings_max.npz", **flat_store)


from diffusers import  DDIMScheduler
import torch

from tqdm import  trange

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler,  DPMSolverMultistepScheduler
from diffusers import  StableDiffusionXLPipeline


import yaml
from types import SimpleNamespace
import random
import numpy as np
import json
import os
# add parent path to sys.path to import lora_diffusion
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from lora_diffusion import patch_pipe

os.environ["HF_HUB_OFFLINE"] = "1"

with open("/leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/thesis/seed.json", "r") as file:
        global_seed = json.load(file)["seed"]
random.seed(global_seed)
np.random.seed(global_seed)
torch.manual_seed(global_seed)
torch.cuda.manual_seed_all(global_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def dummy_checker(image, device, dtype):
    return image, None


def load_pipeline(config_path):
    opt = {}
    with open(config_path, 'r') as file:
        opt = yaml.safe_load(file)

    

    # load sd
    opt['weight_dtype'] = torch.float16
    opt = SimpleNamespace(**opt)

    #vae_path = 'madebyollin/sdxl-vae-fp16-fix'
    vae_path = 'madebyollin/sdxl-vae-fp16-fix'
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

    unet_path = opt.sdxl_unet_path
    unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet")
    #unet = UNet2DConditionModel.from_pretrained(unet_path)
    sdxl_model_path = "stabilityai/stable-diffusion-xl-base-1.0" 
    PIPELINE_NAME = StableDiffusionXLPipeline
    pipeline = PIPELINE_NAME.from_pretrained(sdxl_model_path, vae=vae, unet=unet, torch_dtype=opt.weight_dtype)

    pipeline.unet.to(dtype=opt.weight_dtype)
    pipeline.vae.to(dtype=opt.weight_dtype)
    pipeline.text_encoder.to(dtype=opt.weight_dtype)
    # set grad
    # Freeze vae and text_encoder
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)


    pipeline.run_safety_checker = dummy_checker

    if opt.scheduler == "DPM++":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif opt.scheduler == "DDPM":
        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_opt = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_opt["variance_type"] = variance_type

        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config, **scheduler_opt)
    elif opt.scheduler == 'DDIM':
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    pipeline = pipeline.to('cuda')

    # load attention processors
    pipeline.load_lora_weights(opt.checkpoint_path, weight_name="pytorch_lora_weights.safetensors")
    pipeline.fuse_lora()
    for n, p in pipeline.text_encoder.named_parameters():
        if 'lora' in n:
            print('text encoder lora loaded!')
            break

    pipeline = pipeline.to("cuda")
    pipeline.safety_checker = None
    return pipeline, opt.img, opt.batch_size

def main_comat(pipeline, prompts, img_size, n_iter=1, batch_size=4):
    """
    Generate images using Stable Diffusion with batched prompts.
    
    Args:
        pipeline: The StableDiffusion pipeline.
        prompts (List[str]): A list of text prompts.
        img_size (int): Image height and width.
        n_iter (int): Number of sampling iterations.
        batch_size (int): Number of prompts per batch.
        save_dir (str, optional): Directory to save output images.
    
    Returns:
        List[Image]: Generated images.
    """


    images = []

    with torch.no_grad():
        for n in trange(n_iter, desc="Sampling Iterations"):
            generator = torch.Generator(device="cuda").manual_seed(global_seed)

            for i in trange(0, len(prompts), batch_size, desc="Batches"):
                batch_prompts = prompts[i:i + batch_size]
                
                result = pipeline(
                    batch_prompts,
                    height=img_size,
                    width=img_size,
                    generator=generator
                )

                for j, image in enumerate(result.images):
                    images.append(image)


    return images

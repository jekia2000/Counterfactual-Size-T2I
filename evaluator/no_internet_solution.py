from diffusers import  DDIMScheduler
import torch

from tqdm import  trange

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler,  DPMSolverMultistepScheduler
from diffusers import  StableDiffusionXLPipeline


import yaml
from types import SimpleNamespace
import open_clip


"""
vae_path = 'madebyollin/sdxl-vae-fp16-fix'
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

unet = UNet2DConditionModel.from_pretrained("/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/comat/sdxl/unet_ft100", subfolder="unet")
sdxl_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
PIPELINE_NAME = StableDiffusionXLPipeline
pipeline = PIPELINE_NAME.from_pretrained(sdxl_model_path, vae=vae, unet=unet, torch_dtype=torch.float16)
pipeline.load_lora_weights("/leonardo_work/EUHPC_B22_061/aleksa_files_by_aleksa/downloaded_work/resource/comat/sdxl/lora", weight_name="pytorch_lora_weights.safetensors")
pipeline.fuse_lora()
"""

clip_model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"
open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained, device='cpu')
self.clip_model.eval()

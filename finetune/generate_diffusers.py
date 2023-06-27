import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
from networks.lora import LoRAModule, create_network_from_weights
from safetensors.torch import load_file

# if the ckpt is CompVis based, convert it to Diffusers beforehand with tools/convert_diffusers20_original_sd.py. See --help for more details.

model_id_or_dir = '/fsx/data/models/stable-diffusion_1-5/'
device = "cuda"

# create pipe
print(f"creating pipe from {model_id_or_dir}...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(model_id_or_dir, revision="fp16", vae=vae, torch_dtype=torch.float16)
pipe = pipe.to(device)
vae = pipe.vae
text_encoder = pipe.text_encoder
unet = pipe.unet
pipe.safety_checker = None

generator = torch.Generator(device='cpu')

scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler

# load lora networks
print(f"loading lora networks...")

lora_path1 = '/fsx/data/stable_diffusion/ft_humans/run/output/unsplash-step00004500.safetensors'
sd = load_file(lora_path1)   # If the file is .ckpt, use torch.load instead.
network1, sd = create_network_from_weights(1., None, vae, text_encoder, unet, sd)
network1.apply_to(text_encoder, unet)
network1.load_state_dict(sd)
network1.to(device, dtype=torch.float16)

# # You can merge weights instead of apply_to+load_state_dict. network.set_multiplier does not work
# network.merge_to(text_encoder, unet, sd)

# lora_path2 = r"lora2.safetensors"
# sd = load_file(lora_path2) 
# network2, sd = create_network_from_weights(0.7, None, vae, text_encoder,unet, sd)
# network2.apply_to(text_encoder, unet)
# network2.load_state_dict(sd)
# network2.to(device, dtype=torch.float16)

# lora_path3 = r"lora3.safetensors"
# sd = load_file(lora_path3)
# network3, sd = create_network_from_weights(0.5, None, vae, text_encoder, unet, sd)
# network3.apply_to(text_encoder, unet)
# network3.load_state_dict(sd)
# network3.to(device, dtype=torch.float16)

# prompts
prompt = "man wearing black shirt in front of a black wall in the dark"
# extra_prompt = 'coat, sexy, step, single, hair, posing, graceful, length, stylish, gorgeous, cool, beauty, pose, elegant'
# prompt = f'{prompt}, {extra_prompt}'
negative_prompt = ''
negative_prompt = "low quality, worst quality, bad anatomy, bad composition, poor, low effort"

# execute pipeline
print("generating image...")
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, negative_prompt=negative_prompt, height=768, width=512,
                 num_inference_steps=25).images[0]
    image = pipe(prompt, guidance_scale=7.5, negative_prompt=negative_prompt, height=768, width=512,
                 generator=generator.manual_seed(1), num_inference_steps=25).images[0]

# if not merged, you can use set_multiplier
# network1.set_multiplier(0.8)
# and generate image again...

# save image
image.save(r"by_diffusers..png")
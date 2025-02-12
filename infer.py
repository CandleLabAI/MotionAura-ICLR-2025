from diffusers import MotionAuraPipeline
import torch
from diffusers.utils import export_to_video

device = torch.device("cuda:2")

pipe = MotionAuraPipeline.from_pretrained(
    "onkarsus13/MotionAura-L",
    torch_dtype=torch.bfloat16                                    
    )

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.to(device)

prompt = f"""
Create a graphic novel-style video featuring a group of adventurers exploring an ancient, magical temple.
The camera captures their journey through booby-trapped corridors and mystical chambers.
The scene is filled with intricate, hand-drawn details, vibrant colors, and dramatic lighting.
"""
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49, # increase it according to your compute
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
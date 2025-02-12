from diffusers import AutoencoderKLMotionAura, MotionAuraTransformer3DModel, MotionAuraPipeline

import torch
from diffusers.utils import export_to_video

device = torch.device("cuda:2")
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
# vae = MotionAuraTransformer3DModel.from_pretrained("/data2/onkar/video_diffusion_weights/models--THUDM--CogVideoX-5b/snapshots/8d6ea3f817438460b25595a120f109b88d5fdfad/transformer/")

pipe = MotionAuraPipeline.from_pretrained(
    "/data2/onkar/video_diffusion_weights/models--THUDM--CogVideoX-5b/snapshots/8d6ea3f817438460b25595a120f109b88d5fdfad",
    torch_dtype=torch.bfloat16                                    
    )

pipe.to(device)
# pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()


video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)


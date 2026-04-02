# Runbook: python stream_vae_decode.py 
import torch
from einops import rearrange
from torchvision.io import write_video

latents = torch.load('/mnt/bn/foundation-ads-opt/fangzhou.ai/video_latent/wan_vae/sample.pt', weights_only=True) # B T C H W
latents = latents.cuda()

from utils.wan_wrapper import WanVAEWrapper

vae = WanVAEWrapper("/mnt/bn/foundation-ads-opt/fangzhou.ai/model_ckpt/Wan2.1-T2V-1.3B")
vae = vae.eval().cuda()
vae.to(torch.bfloat16)

# One-pass decode
video = vae.decode_to_pixel(latents, use_cache=False)
video = (video * 0.5 + 0.5).clamp(0, 1)
vae.model.clear_cache()
video = rearrange(video, 'b t c h w -> b t h w c')
write_video("baseline_decode.mp4", 255.0 * video.squeeze(), fps=16)

# Stream style decode
num_frames = latents.shape[1]
frame_per_block = 3
block_videos = []
for idx in range(0, num_frames, frame_per_block):
    block_latents = latents[:, idx:idx+frame_per_block]
    block_video = vae.decode_to_pixel(block_latents, use_cache=True)
    block_video = (block_video * 0.5 + 0.5).clamp(0, 1)
    block_video = rearrange(block_video, 'b t c h w -> b t h w c')
    block_videos.append(block_video)
    write_video(f"stream_decode_{idx}.mp4", 255.0 * block_video.squeeze(), fps=16)
vae.model.clear_cache()

# Verify
block_videos = torch.cat(block_videos, dim=1)
write_video("concated_decode.mp4", 255.0 * block_videos.squeeze(), fps=16)

assert torch.allclose(video, block_videos, atol=1e-2), "Video and block videos are not close"
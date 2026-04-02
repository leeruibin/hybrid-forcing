from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional, Any
import math
import torch
import torch.distributed as dist


class HybridForcingReflowTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 image_or_video_shape: List[int],
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 kv_window_size: int = -1,
                 context_noise: int = 0,
                 use_cross_kv_cache: bool = False,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        self.use_cross_kv_cache = use_cross_kv_cache

        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = self.generator.model.config.num_layers
        self.frame_seq_length = math.prod(image_or_video_shape[-2:]) // math.prod(self.generator.model.patch_size[-2:])
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache_clean = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.num_max_frames = num_max_frames
        if kv_window_size < 0:
            self.kv_cache_size = num_max_frames * self.frame_seq_length
        else:
            self.kv_cache_size = kv_window_size * self.frame_seq_length
    
    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def generate_and_sync_random(self,device="cuda"):
        if dist.get_rank() == 0:
            # 0-1 之间的小数
            rand_val = torch.rand(1, device=device)
        else:
            rand_val = torch.empty(1, device=device)

        # 从 rank 0 广播到所有 rank
        dist.broadcast(rand_val, src=0)

        return rand_val.item()

    def inference(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:

        assert hasattr(self, "reflow_all_step0"), "HybridForcingTrainingPipeline must set reflow_all_step0 [True/False] before inference"  

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        output_step_0 = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_linear_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        
        if self.use_cross_kv_cache:
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
        else:
            # Comment out the _initialize_crossattn_cache to remove cross-attention cache
            self.crossattn_cache = None # train without cross-attention cache

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21
        if exit_flags[0] != 0 and not self.reflow_all_step0:
            output_step_0 = -1
        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # run one step to update global kv state
            with torch.no_grad():
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * self.denoising_step_list[0]
                self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    global_linear_state=self.global_linear_state,
                    current_start=current_start_frame * self.frame_seq_length,
                    is_first_step=True,
                )

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0]) or index == 0
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache_clean,
                            crossattn_cache=self.crossattn_cache,
                            global_linear_state=self.global_linear_state,
                            current_start=current_start_frame * self.frame_seq_length,
                            is_first_step=False,
                        )
                        if index < num_denoising_steps - 1:
                            next_timestep = self.denoising_step_list[index + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache_clean,
                                crossattn_cache=self.crossattn_cache,
                                global_linear_state=self.global_linear_state,
                                current_start=current_start_frame * self.frame_seq_length,
                                is_first_step=False,
                            )
                    else:
                        if self.reflow_all_step0:
                            # This is for calculate gradient
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache_clean,
                                crossattn_cache=self.crossattn_cache,
                                global_linear_state=self.global_linear_state,
                                current_start=current_start_frame * self.frame_seq_length,
                                is_first_step=False,
                            )

                            if index == 0:
                                output_step_0[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
                        else:
                            # only index == 0 we calculate reflow loss
                            if index == exit_flags[0]:
                                # This is for calculate gradient
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache_clean,
                                    crossattn_cache=self.crossattn_cache,
                                    global_linear_state=self.global_linear_state,
                                    current_start=current_start_frame * self.frame_seq_length,
                                    is_first_step=False,
                                )

                                if index == 0:
                                    output_step_0[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
                            else:
                                with torch.no_grad():
                                    _, denoised_pred = self.generator(
                                        noisy_image_or_video=noisy_input,
                                        conditional_dict=conditional_dict,
                                        timestep=timestep,
                                        kv_cache=self.kv_cache_clean,
                                        crossattn_cache=self.crossattn_cache,
                                        global_linear_state=self.global_linear_state,
                                        current_start=current_start_frame * self.frame_seq_length,
                                        is_first_step=False,
                                    )
                        
                    if index == exit_flags[0]:
                        break
            # TODO fold Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    updating_cache=True,
                    global_linear_state=self.global_linear_state,
                    is_first_step=False,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # if noise.device == torch.device("cuda:0"):
        #     print(f"current exit flag is {exit_flags[0]}, timestep is {self.denoising_step_list[exit_flags[0]]}")

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, output_step_0, denoised_timestep_from, denoised_timestep_to

    # TODO for sp adaptation 
    def eval_inference_with_self_forcing(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            conditional_dict: Any = None,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_linear_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        if self.use_cross_kv_cache:
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
        else:
            # Comment out the _initialize_crossattn_cache to remove cross-attention cache, for sp
            self.crossattn_cache = None # train without cross-attention cache

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                
                is_first_step = index == 0
                if index < num_denoising_steps - 1:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache_clean,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            global_linear_state=self.global_linear_state,
                            is_first_step=is_first_step,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache_clean,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            global_linear_state=self.global_linear_state,
                            is_first_step=is_first_step,
                        )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    updating_cache=True,
                    global_linear_state=self.global_linear_state,
                    is_first_step=False,
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        # video = self.vae.decode_to_pixel(output, use_cache=False).cpu()
        # video = (video * 0.5 + 0.5).clamp(0, 1)

        return output

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_clean = []
        head_dim = self.generator.model.config.dim // self.generator.model.config.num_heads
        for _ in range(self.num_transformer_blocks):
            # TODO hard coding now
            kv_cache_clean.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, self.generator.model.config.num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, self.generator.model.config.num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })

        self.kv_cache_clean = kv_cache_clean  # always store the clean cache

    def _initialize_linear_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_clean = []
        head_dim = self.generator.model.config.dim // self.generator.model.config.num_heads
        for _ in range(self.num_transformer_blocks):
            # TODO hard coding now
            kv_cache_clean.append({
                "global_linear_kmean": torch.zeros([batch_size, self.generator.model.config.num_heads, head_dim, 1], dtype=dtype, device=device),
                "global_linear_seqlen": torch.tensor([0], dtype=torch.long, device=device),
                "global_linear_kv": torch.zeros([batch_size, self.generator.model.config.num_heads, head_dim, head_dim], dtype=dtype, device=device),
            })

        self.global_linear_state = kv_cache_clean  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []
        head_dim = self.generator.model.config.dim // self.generator.model.config.num_heads
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, self.generator.model.config.text_len, self.generator.model.config.num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.generator.model.config.text_len, self.generator.model.config.num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
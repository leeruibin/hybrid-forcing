import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD
# from model.dmd_mix import DMDLinear, DMDLinearMXIT2V
from model.dmd_reflow import ReflowDMDLinear
import torch
import time
import os
import wandb
from einops import rearrange
from torchvision.io import write_video

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.max_train_step = getattr(config, "max_train_steps", None)
        self.validation_step = getattr(config, "validation_step", -1)
        validation_prompt_path = getattr(config, "validation_prompt_path", "prompts/example_prompts.txt")
        with open(validation_prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]
        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job(sp_size=config.sp_size)
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.global_rank = global_rank

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        set_seed(config.seed) # For sp we need to enfoce the same seed across all GPUs

        self.output_path = config.logdir

        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            OmegaConf.save(config=self.config, f=f"{self.output_path}/config.yaml")
            if not self.disable_wandb:
                wandb.init(
                    config=OmegaConf.to_container(config, resolve=True),
                    mode="online",
                    project=config.wandb_project,
                    name=config.logdir,
                    dir=config.wandb_save_dir
                )

        

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "dmd_reflow":
            self.model = ReflowDMDLinear(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "generator_cpu_offload", False),
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "real_score_cpu_offload", False),
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "fake_score_cpu_offload", False),
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            fused=True,
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay,
            fused=True,
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, 
                                 shuffle=True, drop_last=True)
        dataloader_num_worker = getattr(config,"dataloader_num_worker",8)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=dataloader_num_worker)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu", weights_only=True)
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                state_dict = state_dict["generator_ema"]
            missing, unexpected = self.model.generator.load_state_dict(
                state_dict, strict=False
            )
            if dist.get_rank() == 0:
                print(f"Missing keys: {missing}")
                print(f"Unexpected keys: {unexpected}")

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step and self.generator_ema is not None:
            ema_state_dict = {
                "generator_ema": self.generator_ema.state_dict(),
            }
            generator_state_dict = {
                "generator": generator_state_dict,
            }
        else:
            ema_state_dict = None
            generator_state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            if ema_state_dict is not None:
                torch.save(ema_state_dict, os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}", "ema_model.pt"))
            torch.save(generator_state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "ema_model.pt, model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            # torch.cuda.empty_cache()
            self.model.generator.model.requires_grad_(True)
            self.model.fake_score.model.requires_grad_(False)
            # if self.step > 2:
            #     for item in self.model.inference_pipeline.kv_cache_clean:
            #         item["fwd"] = True
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
            for item in self.model.inference_pipeline.kv_cache_clean:
                item["fwd"] = False
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        self.model.generator.model.requires_grad_(False)
        self.model.fake_score.model.requires_grad_(True)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    @torch.no_grad()
    def generate_video(self, pipeline, prompts, image=None):
        if not isinstance(prompts, list):
            prompts = [prompts]
        batch_size = len(prompts)
        # TODO hard coding now, can be control in the config file
        num_inference_frames = 126
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=torch.bfloat16)
            # Encode the input image as the first latent
            initial_latent = self.model.vae.encode_to_latent(image).to(device=self.device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_inference_frames - 1, *self.config.image_or_video_shape[-3:]],
                device=self.device,
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_inference_frames, *self.config.image_or_video_shape[-3:]],
                device=self.device,
                dtype=self.dtype
            )
        assert hasattr(pipeline, "eval_inference_with_self_forcing"), "pipeline must have eval_inference_with_self_forcing method"
        conditional_dict = self.model.text_encoder(text_prompts=prompts)
        output_latent = pipeline.eval_inference_with_self_forcing(
            noise=sampled_noise,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )
        video = self.model.vae.decode_to_pixel(output_latent, use_cache=False).cpu()
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video = 255.0 * video
        self.model.vae.model.clear_cache()
        return video, num_inference_frames

    def validation(self):
        if self.model.inference_pipeline is None:
            self.model._initialize_inference_pipeline()
        torch.cuda.empty_cache()
        local_prompts = [self.prompt_list[i] for i in range(self.global_rank, len(self.prompt_list), self.world_size)]
        all_videos = []
        for input_prompt in local_prompts:
            video, num_inference_frame = self.generate_video(self.model.inference_pipeline, input_prompt)
            all_videos.append([video,input_prompt])

        torch.cuda.empty_cache()
        # 多进程：收集所有 rank 的 all_video 到 rank 0
        # gathered_videos = [None for _ in range(self.world_size)]

        all_save_path = []
        all_captions = []
        output_folder = os.path.join(self.output_path, "eval_videos")
        os.makedirs(output_folder, exist_ok=True)

        # gather_object 会把每个 rank 的 all_video 收集到 rank 0 的 gathered_videos 列表中
        # dist.gather_object(all_videos, gathered_videos if self.global_rank == 0 else None, dst=0)
        
        # for src_rank, video_list in enumerate(all_videos):
        for idx, (video_tensor,eval_prompt) in enumerate(all_videos):
            # 可选：检查 tensor 是否在 [0,1] 或 [0,255]
            # wandb.Video 自动处理，但最好归一化到 [0,1] 或 [0,255] 并转 uint8
            if video_tensor.dtype == torch.float32:
                if video_tensor.max() <= 1.0:
                    video_tensor = (video_tensor * 255).clamp(0, 255).to(torch.uint8)
            
            output_path = os.path.join(output_folder, f"validation_step_{self.step}_{eval_prompt[:25]}.mp4")
            write_video(output_path, video_tensor[0], fps=16)
            all_save_path.append(output_path)
            all_captions.append(eval_prompt)
        
        gathered_all_save_path = [None for _ in range(self.world_size)]
        dist.gather_object(all_save_path, gathered_all_save_path if self.global_rank == 0 else None, dst=0)
        all_output_save_path = []
        if self.global_rank == 0:
            for item in gathered_all_save_path:
                all_output_save_path.extend(item)
        

        gathered_all_captions = [None for _ in range(self.world_size)]
        dist.gather_object(all_captions, gathered_all_captions if self.global_rank == 0 else None, dst=0)
        all_output_captions = []
        if self.global_rank == 0:
            for item in gathered_all_captions:
                all_output_captions.extend(item)

        if self.global_rank == 0:
            if not self.disable_wandb:
                logs = {
                    f"validation_videos_frame_{num_inference_frame}": [
                        wandb.Video(filename, caption=caption)
                        for filename, caption in zip(
                            all_output_save_path, all_output_captions, strict=True)
                    ]
                }
                wandb.log(logs, step=self.step)
        del all_videos
        torch.cuda.empty_cache()

    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            if self.validation_step > 0 and self.step % self.validation_step == 0:
                self.validation()
            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "reflow_loss": generator_log_dict.get("reflow_loss", 0),
                            "dmd_loss": generator_log_dict.get("dmd_loss", 0),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)

                    print(
                        f"Step {self.step} | "
                        f"Iteration time: {current_time - self.previous_time:.2f} seconds | "
                    )
                    self.previous_time = current_time
            
            # End train
            if self.max_train_step is not None and self.step > self.max_train_step:
                break
        # save the final model
        torch.cuda.empty_cache()
        self.save()
        torch.cuda.empty_cache()
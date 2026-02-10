import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import Transformer2DModel

from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict

# from diffusers.loaders import FromCkptMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
from tqdm import tqdm
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from loguru import logger


class DiffTipeline(StableDiffusionPipeline):
    def __init__(
        self, reward_model, scheduler: DDIMScheduler, target=1, target_guidance=100
    ):  # ,scheduler: DDIMScheduler
        self.scheduler = scheduler
        self.target = target
        self.target_guidance = target_guidance
        self.reward_model = reward_model
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    @torch.enable_grad()
    def compute_gradient(self, latents, target=None, step=None):
        # latents shape is usually [Batch, Seq, Dim]
        if target is None:
            target = torch.FloatTensor([[self.target]]).to(latents.device)
            target = target.repeat(latents.shape[0], 1)

        # [Core Modification]: Delete the line below!
        # Do not manually swap dimension 0 and 1, keep Batch at dimension 0
        # latents = latents.permute(1, 0, 2).contiguous()

        latents.requires_grad_(True)

        # Now latents is still [Batch, Seq, Dim]
        out = self.reward_model.evaluate(latents)

        # Calculate Loss, where out should be [Batch, 1], target should be [Batch, 1]
        l2_error = 0.5 * torch.nn.MSELoss()(out, target)

        self.reward_model.zero_grad()
        l2_error.backward()

        # The returned gradient shape will be consistent with the input latents [Batch, Seq, Dim]
        return latents.grad.clone(), out

    @torch.no_grad()
    def __call__(
        self,
        diffusion_model: torch.nn.Module,
        shape: Union[List[int], Tuple[int]],
        cond: Optional[torch.FloatTensor],
        steps: int,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        use_reward: int = 0,
        reward_target=None,
        target=1,
        guidance=1,
        generator: Optional[torch.Generator] = None,
        device: torch.device = "cuda:0",
        disable_prog: bool = True,
        cfg_condition: Optional[torch.FloatTensor] = None,
        cfg_guidance_scale: float = 2.0,
    ):

        assert steps > 0, f"{steps} must > 0."

        # CFG guidance: if cfg_condition is provided and cfg_guidance_scale > 1
        do_cfg_guidance = cfg_condition is not None and cfg_guidance_scale > 1.0
        do_classifier_free_guidance = guidance_scale > 1.0  # False

        # init latents
        if cond is not None:
            bsz = cond.shape[0]
            _device = cond.device
            _dtype = cond.dtype
        elif cfg_condition is not None:
            bsz = cfg_condition.shape[0]
            _device = cfg_condition.device
            _dtype = cfg_condition.dtype
        else:
            # Unconditional mode: infer batch size from shape
            bsz = shape[0]
            _device = device
            _dtype = torch.float32

        if do_classifier_free_guidance:
            bsz = bsz // 2

        latents = torch.randn(
            (shape),
            generator=generator,
            device=_device,
            dtype=_dtype,
        )
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(steps, device=device)
        timesteps = self.scheduler.timesteps.to(device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        if eta != 0.0:
            extra_step_kwargs = {"eta": eta, "generator": generator}
        else:
            extra_step_kwargs = {"generator": generator}
        latent_list = []
        # Only process classifier free guidance when cond is not None
        if do_classifier_free_guidance and cond is not None:
            un_cond = torch.zeros_like(cond).float()
            cond = torch.cat([un_cond, cond], dim=0)

        for i, t in enumerate(
            tqdm(timesteps, disable=disable_prog, desc="DDIM Sampling:", leave=False)
        ):
            # expand the latents if we are doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            # predict the noise residual
            timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
            timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
            # cond can be None (unconditional generation)
            noise_pred = diffusion_model.forward(
                latent_model_input, timestep_tensor, cond=cond
            )
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            if use_reward:
                # print('use_reward:', use_reward)
                sqrt_1minus_alpha_t = (1 - self.scheduler.alphas_cumprod[t]) ** 0.5
                computed_gradient, eva_out = self.compute_gradient(latents, step=i)
                noise_pred += (
                    sqrt_1minus_alpha_t * self.target_guidance * computed_gradient
                )
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
            latent_list.append(latents.prev_sample)  # pred_original_sample
            latents = latents.prev_sample
        # print(latents[0])
        if use_reward:
            return latents, latent_list, eva_out
        return latents, latent_list, None


class DiffTipeline_old(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                ############################################################
                ############################################################
                ## Guided Diffusion Modification ##

                ## grad = nabla_x 0.5 * || y - mu(x) ||^2
                ## nabla_x log p_t (y|x_t) = - [1/sigma^2] * grad

                ## For DDIM scheduler,
                ## modified noise = original noise - sqrt( 1-alpha_t ) * (nabla_x log p_t (y|x_t)) ,
                ## see eq(14) of http://arxiv.org/abs/2105.05233

                ## self.target_guidance <---> 1 / sigma^2
                ## self.target  <---> y

                target = torch.FloatTensor([[self.target]]).to(latents.device)
                target = target.repeat(batch_size * num_images_per_prompt, 1)
                sqrt_1minus_alpha_t = (1 - self.scheduler.alphas_cumprod[t]) ** 0.5
                noise_pred += (
                    sqrt_1minus_alpha_t
                    * self.target_guidance
                    * self.compute_gradient(latents, target=target)
                )

                ############################################################
                ############################################################

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            #############################################
            ## Disabled for correct evaluation of the reward
            #############################################
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            #############################################
            ## Disabled for correct evaluation of the reward
            #############################################
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        ##############
        has_nsfw_concept = False
        ##############

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def setup_reward_model(self, reward_model):
        self.reward_model = reward_model
        self.reward_model.requires_grad_(False)
        self.reward_model.eval()

    def set_target(self, target):
        self.target = target

    def set_guidance(self, guidance):
        self.target_guidance = guidance

    @torch.enable_grad()
    def compute_gradient(self, latent, target):
        latent.requires_grad_(True)
        out = self.reward_model(latent)
        l2_error = 0.5 * torch.nn.MSELoss()(out, target)
        self.reward_model.zero_grad()
        l2_error.backward()
        return latent.grad.clone()

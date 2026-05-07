# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .sd3_sde_with_logprob import sde_step_with_logprob,ode_step_with_logprob
from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from .solver import run_sampling
# @torch.no_grad()
# def pipeline_with_logprob(
#     self,
#     prompt: Union[str, List[str]] = None,
#     prompt_2: Optional[Union[str, List[str]]] = None,
#     prompt_3: Optional[Union[str, List[str]]] = None,
#     height: Optional[int] = None,
#     width: Optional[int] = None,
#     num_inference_steps: int = 28,
#     sigmas: Optional[List[float]] = None,
#     guidance_scale: float = 7.0,
#     negative_prompt: Optional[Union[str, List[str]]] = None,
#     negative_prompt_2: Optional[Union[str, List[str]]] = None,
#     negative_prompt_3: Optional[Union[str, List[str]]] = None,
#     num_images_per_prompt: Optional[int] = 1,
#     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#     latents: Optional[torch.FloatTensor] = None,
#     prompt_embeds: Optional[torch.FloatTensor] = None,
#     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#     pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#     negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#     output_type: Optional[str] = "pil",
#     joint_attention_kwargs: Optional[Dict[str, Any]] = None,
#     clip_skip: Optional[int] = None,
#     callback_on_step_end_tensor_inputs: List[str] = ["latents"],
#     max_sequence_length: int = 256,
#     skip_layer_guidance_scale: float = 2.8,
#     noise_level: float = 0.7,
#     return_prev_sample_mean: bool = False
# ):
#     height = height or self.default_sample_size * self.vae_scale_factor
#     width = width or self.default_sample_size * self.vae_scale_factor

#     # 1. Check inputs. Raise error if not correct
#     self.check_inputs(
#         prompt,
#         prompt_2,
#         prompt_3,
#         height,
#         width,
#         negative_prompt=negative_prompt,
#         negative_prompt_2=negative_prompt_2,
#         negative_prompt_3=negative_prompt_3,
#         prompt_embeds=prompt_embeds,
#         negative_prompt_embeds=negative_prompt_embeds,
#         pooled_prompt_embeds=pooled_prompt_embeds,
#         negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
#         callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
#         max_sequence_length=max_sequence_length,
#     )

#     self._guidance_scale = guidance_scale
#     self._skip_layer_guidance_scale = skip_layer_guidance_scale
#     self._clip_skip = clip_skip
#     self._joint_attention_kwargs = joint_attention_kwargs
#     self._interrupt = False

#     # 2. Define call parameters
#     if prompt is not None and isinstance(prompt, str):
#         batch_size = 1
#     elif prompt is not None and isinstance(prompt, list):
#         batch_size = len(prompt)
#     else:
#         batch_size = prompt_embeds.shape[0]

#     device = self._execution_device

#     lora_scale = (
#         self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
#     )
#     (
#         prompt_embeds,
#         negative_prompt_embeds,
#         pooled_prompt_embeds,
#         negative_pooled_prompt_embeds,
#     ) = self.encode_prompt(
#         prompt=prompt,
#         prompt_2=prompt_2,
#         prompt_3=prompt_3,
#         negative_prompt=negative_prompt,
#         negative_prompt_2=negative_prompt_2,
#         negative_prompt_3=negative_prompt_3,
#         do_classifier_free_guidance=self.do_classifier_free_guidance,
#         prompt_embeds=prompt_embeds,
#         negative_prompt_embeds=negative_prompt_embeds,
#         pooled_prompt_embeds=pooled_prompt_embeds,
#         negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
#         device=device,
#         clip_skip=self.clip_skip,
#         num_images_per_prompt=num_images_per_prompt,
#         max_sequence_length=max_sequence_length,
#         lora_scale=lora_scale,
#     )
#     if self.do_classifier_free_guidance:
#         prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
#         pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

#     # 4. Prepare latent variables
#     num_channels_latents = self.transformer.config.in_channels
#     latents = self.prepare_latents(
#         batch_size * num_images_per_prompt,
#         num_channels_latents,
#         height,
#         width,
#         prompt_embeds.dtype,
#         device,
#         generator,
#         latents,
#     ).float()

#     # 5. Prepare timesteps
#     scheduler_kwargs = {}
#     timesteps, num_inference_steps = retrieve_timesteps(
#         self.scheduler,
#         num_inference_steps,
#         device,
#         sigmas=sigmas,
#         **scheduler_kwargs,
#     )
#     num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
#     self._num_timesteps = len(timesteps)

#     # 6. Prepare image embeddings
#     all_latents = [latents]
#     all_log_probs = []
#     all_prev_latents_mean = []

#     # 7. Denoising loop
#     with self.progress_bar(total=num_inference_steps) as progress_bar:
#         for i, t in enumerate(timesteps):
#             if self.interrupt:
#                 continue

#             # expand the latents if we are doing classifier free guidance
#             latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
#             # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
#             timestep = t.expand(latent_model_input.shape[0])
#             noise_pred = self.transformer(
#                 hidden_states=latent_model_input,
#                 timestep=timestep,
#                 encoder_hidden_states=prompt_embeds,
#                 pooled_projections=pooled_prompt_embeds,
#                 joint_attention_kwargs=self.joint_attention_kwargs,
#                 return_dict=False,
#             )[0]
#             # noise_pred = noise_pred.to(prompt_embeds.dtype)
#             # perform guidance
#             if self.do_classifier_free_guidance:
#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 # noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
#                 import random
#                 noise_pred = noise_pred_uncond + random.uniform(1.0, 9.0) * (noise_pred_text - noise_pred_uncond)
                
#             latents_dtype = latents.dtype

#             latents, log_prob, prev_latents_mean, std_dev_t = ode_step_with_logprob(
#                 self.scheduler, 
#                 noise_pred.float(), 
#                 noise_pred_text.float(),
#                 t.unsqueeze(0), 
#                 latents.float(),
#                 noise_level=noise_level,
#             )
            
#             all_latents.append(latents)
#             all_log_probs.append(log_prob)
#             all_prev_latents_mean.append(prev_latents_mean)
#             # if latents.dtype != latents_dtype:
#             #     latents = latents.to(latents_dtype)
            
#             # call the callback, if provided
#             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
#                 progress_bar.update()

#     latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
#     latents = latents.to(dtype=self.vae.dtype)
#     image = self.vae.decode(latents, return_dict=False)[0]
#     image = self.image_processor.postprocess(image, output_type=output_type)

#     # Offload all models
#     self.maybe_free_model_hooks()

#     if return_prev_sample_mean:
#         return image, all_latents, all_log_probs, all_prev_latents_mean
#     return image, all_latents, all_log_probs
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

from diffusers import StableDiffusion3Pipeline

@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
    deterministic: bool = False,
    solver: str = "flow",
    model_type: str = "sd3",
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    assert model_type in ["sd3", "flux"]
    flux = model_type == "flux"
    # 1. Check inputs. Raise error if not correct
    if not flux:
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
    else:
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    if not flux:
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    # 4. Prepare latent variables
    if not flux:
        num_channels_latents = self.transformer.config.in_channels
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
    else:
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

    # 5. Prepare timesteps
    if not flux:
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=None,
        )
        self._num_timesteps = len(timesteps)
    else:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

    sigmas = self.scheduler.sigmas.float()

    def v_pred_fn(z, sigma):
        if not flux:
            latent_model_input = torch.cat([z] * 2) if self.do_classifier_free_guidance else z
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = torch.full([latent_model_input.shape[0]], sigma * 1000, device=z.device, dtype=torch.long)
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            latent_model_input = z
            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(latent_model_input.shape[0])
            else:
                guidance = None
            timesteps = torch.full([latent_model_input.shape[0]], sigma, device=z.device, dtype=torch.long)
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
        return noise_pred

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []

    # 7. Denoising loop
    latents, all_latents, all_log_probs = run_sampling(v_pred_fn, latents, sigmas, solver, deterministic, noise_level)

    if flux:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not flux:
        return image, all_latents, all_log_probs
    else:
        return image, all_latents, latent_image_ids, text_ids, all_log_probs
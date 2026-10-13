"""SD3.5-Medium pieces: text encoders, transformer loading with LoRA, decoding.

Everything here works on plain ``[B, 16, H/8, W/8]`` latents and calls the
diffusers ``SD3Transformer2DModel`` directly. The prompt encoder follows the
diffusers SD3 convention that the reference training used: the two CLIP towers
are concatenated on the feature axis, padded up to the T5 width, and prepended
to the T5 sequence, while the two CLIP pooled vectors are concatenated into the
pooled projection.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.image_processor import VaeImageProcessor
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)


def all_linear_module_names(module: torch.nn.Module, prefix: str = "") -> List[str]:
    """Fully-qualified names of every ``nn.Linear`` leaf: the LoRA target set.

    The names have to stay fully qualified. peft matches a target entry by
    suffix, so handing it bare leaf names such as ``"0"`` or ``"linear"`` makes
    it select container blocks it cannot adapt, and injection fails. The
    adapter files on disk list a shortened set because peft minimises it when
    saving, which is a different question from what to pass in.
    """
    names: List[str] = []
    for name, child in module.named_children():
        current = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            names.append(current)
        else:
            names.extend(all_linear_module_names(child, prefix=current))
    return names


class SD3TextEncoders:
    """CLIP-L + CLIP-G + T5-XXL, frozen, producing SD3 joint prompt embeddings."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_t5_length: int = 256,
    ):
        self.device = device
        self.dtype = dtype
        self.max_t5_length = int(max_t5_length)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            os.path.join(model_path, "text_encoder"), torch_dtype=dtype
        ).to(device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            os.path.join(model_path, "text_encoder_2"), torch_dtype=dtype
        ).to(device)
        self.text_encoder_3 = T5EncoderModel.from_pretrained(
            os.path.join(model_path, "text_encoder_3"), torch_dtype=dtype
        ).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(model_path, "tokenizer")
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            os.path.join(model_path, "tokenizer_2")
        )
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(
            os.path.join(model_path, "tokenizer_3")
        )
        for te in (self.text_encoder, self.text_encoder_2, self.text_encoder_3):
            te.requires_grad_(False)
            te.eval()

    def _encode_clip(self, tokenizer, text_encoder, prompts: Sequence[str]):
        ids = tokenizer(
            list(prompts),
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        out = text_encoder(ids, output_hidden_states=True)
        return out.hidden_states[-2], out[0]

    def _encode_t5(self, prompts: Sequence[str]):
        ids = self.tokenizer_3(
            list(prompts),
            padding="max_length",
            max_length=self.max_t5_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder_3(ids)[0]

    @torch.no_grad()
    def encode(self, prompts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prompt_embeds [B, 77 + L_t5, 4096], pooled [B, 2048])``."""
        prompts = ["" if p is None else p for p in prompts]
        clip_l_seq, clip_l_pool = self._encode_clip(
            self.tokenizer, self.text_encoder, prompts
        )
        clip_g_seq, clip_g_pool = self._encode_clip(
            self.tokenizer_2, self.text_encoder_2, prompts
        )
        clip_seq = torch.cat([clip_l_seq, clip_g_seq], dim=-1)
        pooled = torch.cat([clip_l_pool, clip_g_pool], dim=-1)
        t5_seq = self._encode_t5(prompts)
        clip_seq = F.pad(clip_seq, (0, t5_seq.shape[-1] - clip_seq.shape[-1]))
        embeds = torch.cat([clip_seq, t5_seq], dim=1)
        return embeds.to(self.dtype), pooled.to(self.dtype)


def load_transformer(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    merge_lora_path: Optional[str] = None,
    gradient_checkpointing: bool = False,
) -> SD3Transformer2DModel:
    """Load the SD3 transformer, optionally folding one LoRA into the weights.

    ``merge_lora_path`` is applied through peft and then merged away, so the
    returned module is an ordinary ``nn.Module`` with no adapter attached. This
    is how both the frozen expert teachers (base plus their own LoRA) and the
    student's starting point (base plus the initialisation soup) are built.
    """
    transformer = SD3Transformer2DModel.from_pretrained(
        os.path.join(model_path, "transformer"), torch_dtype=dtype
    ).to(device)
    if merge_lora_path:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(
            transformer,
            merge_lora_path,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
        transformer = peft_model.merge_and_unload()
        transformer.to(dtype)
    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    return transformer


def attach_fresh_lora(
    transformer: torch.nn.Module,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    init_lora_weights: str = "gaussian",
) -> torch.nn.Module:
    """Wrap the transformer in a freshly initialised trainable LoRA adapter."""
    import peft

    config = peft.LoraConfig(
        init_lora_weights=init_lora_weights,
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        target_modules=all_linear_module_names(transformer),
    )
    return peft.get_peft_model(transformer, config, autocast_adapter_dtype=False)


def load_lora_adapter(
    transformer: torch.nn.Module,
    adapter_path: str,
    dtype: torch.dtype = torch.bfloat16,
    is_trainable: bool = False,
) -> torch.nn.Module:
    """Attach an existing LoRA checkpoint as a live adapter (not merged)."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        transformer,
        adapter_path,
        autocast_adapter_dtype=False,
        is_trainable=is_trainable,
    )
    model.to(dtype)
    return model


def velocity(
    transformer: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
) -> torch.Tensor:
    """One transformer call.

    ``timestep`` is ``(B,)`` on the scheduler's discrete axis, that is ``sigma``
    scaled to ``[0, 1000]``; callers that hold a sigma multiply before calling,
    which keeps the arithmetic identical to the reference sampler. Conditioning
    tensors are cast to the latent dtype for the same reason.
    """
    dtype = latents.dtype
    return transformer(
        hidden_states=latents,
        timestep=timestep.reshape(-1).to(dtype),
        encoder_hidden_states=prompt_embeds.to(dtype),
        pooled_projections=pooled.to(dtype),
        return_dict=False,
    )[0]


class LatentDecoder:
    """SD3 VAE decode to PIL, kept in fp32 for numerical stability."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.vae = AutoencoderKL.from_pretrained(
            os.path.join(model_path, "vae"), torch_dtype=torch.float32
        ).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)

    @torch.no_grad()
    def to_pil(self, latents: torch.Tensor) -> list:
        """Decode ``[B, 16, h, w]`` latents one sample at a time."""
        sf = self.vae.config.scaling_factor
        shift = getattr(self.vae.config, "shift_factor", 0.0) or 0.0
        images = []
        for i in range(latents.shape[0]):
            lat = latents[i : i + 1].to(self.vae.dtype) / sf + shift
            decoded = self.vae.decode(lat, return_dict=False)[0]
            images.extend(self.image_processor.postprocess(decoded, output_type="pil"))
        return images


def load_scheduler(model_path: str) -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(model_path, "scheduler")
    )

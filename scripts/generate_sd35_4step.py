#!/usr/bin/env python3
"""Four-step sampling from a trained adapter.

The weights are assembled in three layers, in this order: the base SD3.5-Medium
transformer, then the initialisation soup folded into those weights, then the
trained adapter loaded on top and merged away. Loading the trained adapter
before folding the soup would place the low-rank update on the wrong weights, so
the order is load-bearing and matches how the checkpoints were produced.

Sampling uses the inference grid, which is a linspace in the pre-shift space put
through the scheduler's shift, four steps, and no classifier-free guidance,
since the distilled student is trained to be queried that way. Prompts come
from a JSON file: either a list of strings, or a list of objects carrying a
``prompt`` field and optionally an ``id``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stepx_opd.sd3_model import (
    LatentDecoder,
    SD3TextEncoders,
    load_lora_adapter,
    load_scheduler,
    load_transformer,
    velocity,
)
from stepx_opd.schedule import calculate_shift, retrieve_timesteps


def resolve_adapter_dir(path: str) -> str:
    """Accept either an adapter directory or a checkpoint that contains one."""
    nested = os.path.join(path, "transformer")
    if os.path.isfile(os.path.join(nested, "adapter_config.json")):
        return nested
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    raise FileNotFoundError(
        f"no adapter_config.json under {path} or {path}/transformer"
    )


def load_prompt_file(path: str):
    with open(path) as handle:
        raw = json.load(handle)
    items = []
    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            items.append({"id": f"{index:05d}", "prompt": entry})
        else:
            items.append(
                {
                    "id": str(entry.get("id") or entry.get("pid") or f"{index:05d}"),
                    "prompt": entry["prompt"],
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="SD3.5-Medium diffusers directory")
    parser.add_argument(
        "--init_merge",
        default=None,
        help="initialisation soup folded into the base before the adapter",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="trained adapter directory, or a checkpoint containing transformer/",
    )
    parser.add_argument("--prompts", required=True, help="JSON prompt file")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_t5_length", type=int, default=256)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    prompts = load_prompt_file(args.prompts)
    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"[gen] base={args.base}\n[gen] init_merge={args.init_merge}\n"
        f"[gen] adapter={args.adapter}\n[gen] steps={args.steps} cfg={args.cfg} "
        f"size={args.size} seeds={seeds} prompts={len(prompts)}",
        flush=True,
    )

    transformer = load_transformer(
        args.base, device=device, dtype=dtype, merge_lora_path=args.init_merge
    )
    if args.adapter:
        transformer = load_lora_adapter(
            transformer, resolve_adapter_dir(args.adapter), dtype=dtype
        )
        transformer = transformer.merge_and_unload()
        print("[gen] trained adapter merged into the weights", flush=True)
    if hasattr(transformer, "gradient_checkpointing_disable"):
        transformer.gradient_checkpointing_disable()
    transformer.eval()

    text_encoders = SD3TextEncoders(
        args.base, device=device, dtype=dtype, max_t5_length=args.max_t5_length
    )
    decoder = LatentDecoder(args.base, device=device)
    scheduler = load_scheduler(args.base)

    lat = args.size // 8
    mu = calculate_shift(
        (lat // 2) ** 2,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )

    manifest = []
    for item in prompts:
        embeds, pooled = text_encoders.encode([item["prompt"]])
        neg_embeds = neg_pooled = None
        if args.cfg > 1.0:
            neg_embeds, neg_pooled = text_encoders.encode([args.negative_prompt])
        for seed in seeds:
            out_path = os.path.join(args.out_dir, f"{item['id']}_seed{seed}.png")
            if (
                args.skip_existing
                and os.path.exists(out_path)
                and os.path.getsize(out_path) > 64
            ):
                continue

            generator = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                (1, 16, lat, lat), generator=generator, device=device, dtype=dtype
            )
            sigmas = np.linspace(1.0, 1.0 / args.steps, args.steps)
            timesteps, _ = retrieve_timesteps(
                scheduler, args.steps, device, sigmas=sigmas, mu=mu
            )
            scheduler.set_begin_index(0)
            with torch.no_grad():
                for t in timesteps:
                    ts = t.expand(1)
                    v = velocity(transformer, latents, ts, embeds, pooled)
                    if args.cfg > 1.0:
                        v_neg = velocity(
                            transformer, latents, ts, neg_embeds, neg_pooled
                        )
                        v = v_neg + args.cfg * (v - v_neg)
                    latents = scheduler.step(v, t, latents, return_dict=False)[0]
                image = decoder.to_pil(latents)[0]
            image.save(out_path)
            manifest.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "seed": seed,
                    "path": out_path,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "size": args.size,
                    "base": args.base,
                    "init_merge": args.init_merge,
                    "adapter": args.adapter,
                }
            )
            print(f"[gen] {out_path}", flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"[gen] done, {len(manifest)} images", flush=True)


if __name__ == "__main__":
    main()

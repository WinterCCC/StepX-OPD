#!/usr/bin/env python3
"""Four-step distribution matching for SD3.5-Medium with w_b-weighted velocity regression.

One training step runs the following. The student first produces a four-step
deterministic Euler rollout without gradients, which gives the states that all
supervision is attached to. One segment of that rollout is then re-forwarded
with gradients, and a supervision time ``tau`` is drawn inside the segment: the
segment is interpolated at a base time and re-noised forward to ``tau`` under
the exact forward law, so the query point is a genuine sample of the forward
process rather than a grid endpoint. At that point the trainable critic is
updated to regress the student's own clean estimate, and the student is then
pushed along the difference between the frozen teacher's guided velocity and the
critic's velocity, which is the distribution-matching gradient. The teacher is
the expert that owns the source this step's prompts came from, so routing is a
property of the data and needs no cross-rank agreement beyond the shared step
counter. Finally, when the drawn ``tau`` sits in the low-noise half of the
range, the student's velocity at ``tau`` is regressed directly onto the
teacher's, and this variant weights that regression per sample by ``w_b``.

Launch with torchrun; see the README for the exact command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import uuid

os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stepx_opd.data import MultiSourcePromptSampler
from stepx_opd.objective import (
    conditional_is_weight,
    critic_omega,
    pseudo_huber_loss,
    wb_omega,
)
from stepx_opd.sd3_model import (
    SD3TextEncoders,
    attach_fresh_lora,
    load_scheduler,
    load_transformer,
    velocity,
)
from stepx_opd.schedule import (
    build_stratified_random_grid,
    calculate_shift,
    effective_shift,
    flow_matching_renoise_sde,
)

SEED_SELECT_SEGMENT = 11
SEED_TAU = 12345


def log0(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message, flush=True)


def expand_time(t: torch.Tensor) -> torch.Tensor:
    """``(B,) -> (B, 1, 1, 1)`` so a per-sample time broadcasts over a latent."""
    return t.reshape(-1, 1, 1, 1)


def allreduce_mean_grads(params: List[torch.nn.Parameter], world_size: int) -> None:
    """Average gradients across ranks.

    The trainable set is only the LoRA tensors, so an explicit all-reduce is
    cheaper to read than a wrapper and it keeps the two optimisers independent:
    the critic is stepped once before the student in the same iteration, and
    each has its own synchronisation point.
    """
    if world_size <= 1:
        return
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def build_optimizer(params, lr: float, betas, weight_decay: float, warmup_steps: int):
    optimizer = torch.optim.AdamW(
        params, lr=float(lr), betas=tuple(betas), weight_decay=float(weight_decay)
    )
    warmup = max(0, int(warmup_steps))

    def lr_lambda(step: int) -> float:
        if warmup == 0:
            return 1.0
        return min(1.0, float(step) / float(warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def save_adapter(module: torch.nn.Module, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    module.save_pretrained(directory)


TRAIN_STATE_SUBDIR = "training_state"

COMPLETE_MARKER = "complete"

RESUME_FINGERPRINT_IGNORED = frozenset(
    {"output_dir", "run_name", "log_interval", "save_interval", "max_train_steps"}
)


def resume_fingerprint(cfg: Dict) -> str:
    """A digest of every config value a resumed step depends on.

    The per-teacher ``name`` is dropped alongside the five run-level keys,
    because it only labels a log line and a metrics field, while the teacher
    list's order is kept: it is what the source router indexes into.
    """
    values = {k: v for k, v in cfg.items() if k not in RESUME_FINGERPRINT_IGNORED}
    values["teachers"] = [
        {k: v for k, v in entry.items() if k != "name"} for entry in cfg["teachers"]
    ]
    values.setdefault("source_weights", None)
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_state_path(checkpoint_dir: str, rank: int) -> str:
    return os.path.join(checkpoint_dir, TRAIN_STATE_SUBDIR, f"rank_{rank}.pt")


def complete_marker_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, TRAIN_STATE_SUBDIR, COMPLETE_MARKER)


def trainable_tensors(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """The low-rank adapter tensors of one module, keyed by parameter name.

    The trainable set is exactly what carries the run forward, so saving the
    parameters directly is both smaller and more faithful than a second peft
    export: the names come from the live module, which means a restore can
    check them against the module it is loading into.
    """
    return {
        name: param.detach().to("cpu", copy=True)
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def load_trainable_tensors(
    module: torch.nn.Module, saved: Dict[str, torch.Tensor], what: str
) -> None:
    live = {name: p for name, p in module.named_parameters() if p.requires_grad}
    missing = sorted(set(live) - set(saved))
    unexpected = sorted(set(saved) - set(live))
    if missing or unexpected:
        raise RuntimeError(
            f"{what}: the checkpoint's trainable tensors do not match this model; "
            f"{len(missing)} missing (e.g. {missing[:2]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:2]}). "
            "This usually means lora_rank or the target module set changed."
        )
    with torch.no_grad():
        for name, param in live.items():
            tensor = saved[name]
            if tuple(tensor.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"{what}: shape mismatch for {name}, checkpoint "
                    f"{tuple(tensor.shape)} vs model {tuple(param.shape)}"
                )
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))


def rng_state(device: str) -> Dict:
    """Snapshot of the four global random streams.

    No draw in the training loop reads any of them: the grid, the segment
    choice, the initial noise and the three draws around ``tau`` all come from
    generators seeded from the step counter, which is what makes every rank
    agree without a collective. They are saved anyway, so that a future draw
    that does reach for a global stream resumes correctly rather than silently
    restarting from a fresh seed.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state: Dict, device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def save_training_state(
    checkpoint_dir: str,
    rank: int,
    world_size: int,
    global_step: int,
    device: str,
    student,
    critic,
    student_opt,
    critic_opt,
    student_sched,
    critic_sched,
    checkpoint_id: str,
    config_fingerprint: str,
) -> str:
    path = training_state_path(checkpoint_dir, rank)
    payload = {
        "global_step": int(global_step),
        "rank": int(rank),
        "world_size": int(world_size),
        "checkpoint_id": str(checkpoint_id),
        "config_fingerprint": str(config_fingerprint),
        "student": trainable_tensors(student),
        "critic": trainable_tensors(critic),
        "student_opt": student_opt.state_dict(),
        "critic_opt": critic_opt.state_dict(),
        "student_sched": student_sched.state_dict(),
        "critic_sched": critic_sched.state_dict(),
        "rng": rng_state(device),
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_training_state(
    checkpoint_dir: str,
    rank: int,
    world_size: int,
    device: str,
    student,
    critic,
    student_opt,
    critic_opt,
    student_sched,
    critic_sched,
    config_fingerprint: str,
) -> int:
    """Restore one rank's state and return the step the checkpoint ended on.

    Nothing is restored until every rank has accepted its own file. Each rank
    checks the marker, its payload's id, its rank and world size, the step the
    directory name claims and the config digest, then the verdicts are gathered
    so that one rank's rejection stops all of them. Without the gather a healthy
    rank would walk into the first collective of the training loop while another
    had already raised, and the launch would hang instead of failing.
    """
    error = None
    identity = None
    payload = None
    try:
        marker = complete_marker_path(checkpoint_dir)
        if not os.path.isfile(marker):
            raise RuntimeError(
                f"no completion marker at {marker}. One is published only after "
                "every rank has written its state, so its absence means either "
                "a save that was interrupted or a checkpoint written before "
                "saves were marked at all. In both cases the files in the "
                "directory cannot be shown to come from a single save, and a "
                "resume that read them anyway could silently mix two runs; "
                "train forward from a checkpoint this version of the script "
                "wrote. The two adapter directories are untouched either way, "
                "so this checkpoint still loads for generation."
            )
        with open(marker, encoding="ascii") as handle:
            checkpoint_id = handle.read().strip()

        path = training_state_path(checkpoint_dir, rank)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"no training state for rank {rank} at {path}; a checkpoint "
                "carries one file per rank and the launch has "
                f"{world_size} ranks."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        saved_world = int(payload["world_size"])
        if saved_world != int(world_size):
            raise RuntimeError(
                f"this checkpoint was written by a {saved_world}-rank run and "
                f"the launch has {world_size} ranks. The prompt slices and the "
                "gradient average both depend on the rank count, so the run "
                "would not be a continuation; relaunch with the original rank "
                "count."
            )
        step = int(payload["global_step"])
        named_step = os.path.basename(os.path.normpath(checkpoint_dir))
        if (
            not checkpoint_id
            or payload.get("checkpoint_id") != checkpoint_id
            or int(payload["rank"]) != rank
            or named_step != f"checkpoint-step-{step}"
        ):
            raise RuntimeError(
                "this file does not belong to the save the marker describes: "
                f"marker id {checkpoint_id!r}, file id "
                f"{payload.get('checkpoint_id')!r}, file rank "
                f"{payload.get('rank')} against launched rank {rank}, file step "
                f"{step} against directory {named_step!r}. A directory whose "
                "rank files carry different ids holds pieces of two different "
                "saves, which is not a state any run ever had."
            )
        if payload.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError(
                "the config does not match the one that wrote this checkpoint: "
                f"saved digest {payload.get('config_fingerprint')!r} against "
                f"this launch's {config_fingerprint!r}. Most of the config is "
                "rebuilt on resume rather than restored, so a changed value "
                "would quietly alter the objective, the data stream, the frozen "
                "models or a warmup schedule while the run presented itself as "
                "a continuation; the warmup lengths are the plainest case, "
                "because they live in a closure that no scheduler state_dict "
                "serialises. Resume with the config that wrote the checkpoint, "
                "or start a new run under a new run_name. Only output_dir, "
                "run_name, log_interval, save_interval and max_train_steps are "
                "free to change."
            )
        identity = (checkpoint_id, step, int(world_size))
    except Exception as exc:  # noqa: BLE001 - reported collectively below
        error = f"rank {rank}: {type(exc).__name__}: {exc}"

    reports = [(identity, error)]
    if dist.is_initialized():
        reports = [None] * world_size
        dist.all_gather_object(reports, (identity, error))
    failures = [message for _, message in reports if message is not None]
    if failures:
        raise RuntimeError("checkpoint validation failed:\n" + "\n".join(failures))
    if any(seen != reports[0][0] for seen, _ in reports):
        raise RuntimeError(
            "the ranks accepted different saves: "
            + ", ".join(f"rank {i}: {seen}" for i, (seen, _) in enumerate(reports))
        )

    load_trainable_tensors(student, payload["student"], "student")
    load_trainable_tensors(critic, payload["critic"], "critic")
    student_opt.load_state_dict(payload["student_opt"])
    critic_opt.load_state_dict(payload["critic_opt"])
    student_sched.load_state_dict(payload["student_sched"])
    critic_sched.load_state_dict(payload["critic_sched"])
    restore_rng_state(payload["rng"], device)
    return int(payload["global_step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--override",
        default=None,
        help="inline YAML merged over the config file, e.g. "
        "'{max_train_steps: 40, train_batch_size: 4}'",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="a checkpoint-step-<N> directory to continue from; training picks "
        "up at step N+1 and runs to max_train_steps. The directory must hold "
        f"a {TRAIN_STATE_SUBDIR}/ subdirectory with one file per rank, and the "
        "launch must use the same number of ranks that wrote it.",
    )
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    if args.override:
        cfg.update(yaml.safe_load(args.override))
    config_fingerprint = resume_fingerprint(cfg)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    dtype = torch.bfloat16
    resolution = int(cfg["resolution"])
    lat_size = resolution // 8
    num_steps = int(cfg["num_inference_steps"])
    bsz = int(cfg["train_batch_size"])
    max_steps = int(cfg["max_train_steps"])
    teacher_warmup = int(cfg["teacher_warmup_steps"])
    vmse_weight = float(cfg["lownoise_vmse_weight"])
    vmse_start = int(cfg["lownoise_vmse_start_step"])
    tau_threshold = float(cfg["lownoise_tau_threshold"])
    wb_mode = str(cfg["wb_omega_mode"])
    wb_eps = float(cfg["wb_omega_eps"])
    wb_normalize = bool(cfg["wb_omega_normalize"])
    use_wb = bool(cfg["lownoise_vmse_wb"])
    guidance_scale = float(cfg["teacher_guidance_scale"])
    critic_eps = float(cfg["critic_omega_eps"])
    huber_c = float(cfg["pseudo_huber_c_coef"])
    out_dir = os.path.join(cfg["output_dir"], cfg["run_name"])
    ckpt_dir = os.path.join(out_dir, "checkpoint")

    log0(
        "=" * 78
        + f"\n run={cfg['run_name']}  world_size={world_size}"
        + f"  per_rank_batch={bsz}  global_batch={bsz * world_size}"
        + f"\n steps={num_steps}  resolution={resolution}"
        + f"  student_lr={cfg['student_lr']}  critic_lr={cfg['critic_lr']}"
        + f"\n teacher_warmup={teacher_warmup}  teacher_gs={guidance_scale}"
        + f"\n lownoise_vmse: weight={vmse_weight} start={vmse_start}"
        + f" tau<={tau_threshold} wb={use_wb} mode={wb_mode}"
        + f" eps={wb_eps} normalize={wb_normalize}"
        + "\n" + "=" * 78
    )

    base_path = cfg["base_model_path"]
    scheduler = load_scheduler(base_path)
    shift_factor = effective_shift(
        scheduler, calculate_shift((lat_size // 2) ** 2)
    )

    log0(f"[load] text encoders from {base_path}")
    text_encoders = SD3TextEncoders(
        base_path, device=device, dtype=dtype, max_t5_length=int(cfg["max_t5_length"])
    )

    log0(f"[load] student backbone, soup merged from {cfg['init_lora_merge_path']}")
    student_backbone = load_transformer(
        base_path,
        device=device,
        dtype=dtype,
        merge_lora_path=cfg["init_lora_merge_path"],
        gradient_checkpointing=True,
    )
    import copy

    critic_backbone = copy.deepcopy(student_backbone)
    student = attach_fresh_lora(
        student_backbone, int(cfg["lora_rank"]), int(cfg["lora_alpha"])
    )
    critic = attach_fresh_lora(
        critic_backbone, int(cfg["lora_rank"]), int(cfg["lora_alpha"])
    )

    log0("[load] warmup teacher = student backbone with the adapter disabled")

    teachers: List[torch.nn.Module] = []
    teacher_names: List[str] = []
    for entry in cfg["teachers"]:
        log0(f"[load] teacher {entry['name']} from {entry['lora_path']}")
        teachers.append(
            load_transformer(
                base_path,
                device=device,
                dtype=dtype,
                merge_lora_path=entry["lora_path"],
                gradient_checkpointing=False,
            )
        )
        teachers[-1].eval()
        teachers[-1].requires_grad_(False)
        teacher_names.append(entry["name"])
    source_to_teacher = {
        entry["source"]: idx for idx, entry in enumerate(cfg["teachers"])
    }
    teacher_gs = [
        float(entry.get("guidance_scale", guidance_scale)) for entry in cfg["teachers"]
    ]

    sampler = MultiSourcePromptSampler(
        source_files={k: v for k, v in cfg["source_files"].items()},
        source_weights=cfg.get("source_weights"),
        batch_size=bsz,
        rank=rank,
        world_size=world_size,
        seed=int(cfg["data_seed"]),
    )
    log0(f"[data] sources={sampler.sources} pattern={sampler.pattern}")

    student_params = [p for p in student.parameters() if p.requires_grad]
    critic_params = [p for p in critic.parameters() if p.requires_grad]
    log0(
        f"[params] trainable student={sum(p.numel() for p in student_params)/1e6:.1f}M"
        f" critic={sum(p.numel() for p in critic_params)/1e6:.1f}M"
    )
    student_opt, student_sched = build_optimizer(
        student_params,
        cfg["student_lr"],
        cfg["betas"],
        cfg["weight_decay"],
        cfg["warmup_steps"],
    )
    critic_opt, critic_sched = build_optimizer(
        critic_params,
        cfg["critic_lr"],
        cfg["betas"],
        cfg["weight_decay"],
        cfg["critic_warmup_steps"],
    )

    start_step = 0
    if args.resume_from:
        log0(f"[resume] from {args.resume_from}")
        resumed_at = load_training_state(
            args.resume_from,
            rank,
            world_size,
            device,
            student,
            critic,
            student_opt,
            critic_opt,
            student_sched,
            critic_sched,
            config_fingerprint=config_fingerprint,
        )
        start_step = resumed_at + 1
        log0(
            f"[resume] checkpoint ended on step {resumed_at}, continuing at"
            f" {start_step} with student_lr={student_opt.param_groups[0]['lr']:.3e}"
            f" critic_lr={critic_opt.param_groups[0]['lr']:.3e}"
        )
        if start_step >= max_steps:
            log0(
                f"[resume] max_train_steps={max_steps} is already reached;"
                " nothing to do"
            )

    neg_embeds, neg_pooled = text_encoders.encode([cfg["negative_prompt"]] * bsz)

    init_noise_gen = torch.Generator(device=device)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"metrics_rank{rank}.jsonl")
    if args.resume_from and os.path.exists(metrics_path):
        rows: Dict[int, bytes] = {}
        with open(metrics_path, "rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    break
                step = int(json.loads(line)["step"])
                if step < start_step:
                    rows[step] = line
        tmp = f"{metrics_path}.tmp"
        with open(tmp, "wb") as handle:
            for step in sorted(rows):
                handle.write(rows[step])
        os.replace(tmp, metrics_path)
    metrics_file = open(metrics_path, "a" if args.resume_from else "w")

    t_start = time.time()
    for global_step in range(start_step, max_steps):
        batch = sampler.batch(global_step)
        source = batch["source"]
        prompt_embeds, pooled = text_encoders.encode(batch["prompts"])

        timesteps, sigmas = build_stratified_random_grid(
            num_steps, global_step, shift_factor, device
        )

        init_noise_gen.manual_seed(
            int(cfg["noise_seed"]) + global_step * 1000003 + rank
        )
        noise = torch.randn(
            (bsz, 16, lat_size, lat_size),
            generator=init_noise_gen,
            device=device,
            dtype=dtype,
        )

        student.eval()
        intermediates = [noise]
        with torch.no_grad():
            z = noise
            for i in range(num_steps):
                v = velocity(
                    student, z, timesteps[i].expand(bsz), prompt_embeds, pooled
                )
                z = z + (sigmas[i + 1] - sigmas[i]) * v
                intermediates.append(z.detach())

        selected = random.Random(global_step + SEED_SELECT_SEGMENT).randrange(num_steps)
        z_start = intermediates[selected]
        sigma_upper = float(sigmas[selected].item())
        sigma_lower = float(sigmas[selected + 1].item())

        student.train()
        v_selected = velocity(
            student, z_start, timesteps[selected].expand(bsz), prompt_embeds, pooled
        )
        z_end = z_start + (sigma_lower - sigma_upper) * v_selected
        z_end_det = z_end.detach()

        with torch.no_grad():
            if selected + 1 < num_steps:
                v_end = velocity(
                    student,
                    z_end_det,
                    timesteps[selected + 1].expand(bsz),
                    prompt_embeds,
                    pooled,
                )
                x0_target = (z_end_det - sigma_lower * v_end).detach()
            else:
                x0_target = z_end_det

        tau_gen = torch.Generator(device=device)
        tau_gen.manual_seed(global_step + SEED_TAU)
        span = max(sigma_upper - sigma_lower, 1e-6)
        tau_base = (
            sigma_lower
            + torch.rand((bsz,), generator=tau_gen, device=device, dtype=dtype) * span
        )
        interp = (
            (tau_base - sigma_lower) / max(sigma_upper - sigma_lower, 1e-8)
        ).clamp(0.0, 1.0)
        x_base = expand_time(interp) * z_start + (1.0 - expand_time(interp)) * z_end_det
        u_tau = torch.rand((bsz,), generator=tau_gen, device=device, dtype=dtype)
        tau = tau_base + u_tau * (sigma_upper - tau_base).clamp(min=0.0)
        bridge_noise = torch.randn(
            z_end_det.shape, generator=tau_gen, device=device, dtype=z_end_det.dtype
        )
        z_tau = flow_matching_renoise_sde(
            x_base, expand_time(tau_base), expand_time(tau), bridge_noise
        )

        critic.train()
        critic_opt.zero_grad(set_to_none=True)
        v_critic = velocity(critic, z_tau, tau * 1000, prompt_embeds, pooled)
        x0_critic = z_tau - expand_time(tau) * v_critic
        is_weight = conditional_is_weight(z_tau, x_base, x0_target, tau_base, tau)
        c_omega = critic_omega(tau, critic_eps)
        loss_critic = (
            ((x0_critic - x0_target) ** 2)
            * expand_time(is_weight.to(dtype))
            * expand_time(c_omega.to(dtype))
        ).mean()
        loss_critic.backward()
        allreduce_mean_grads(critic_params, world_size)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_params, float(cfg["clip_grad_norm"])
        )
        if torch.isfinite(critic_grad_norm):
            critic_opt.step()
            critic_sched.step()
        critic_opt.zero_grad(set_to_none=True)

        in_warmup = teacher_warmup > 0 and global_step < teacher_warmup
        if in_warmup:
            routed_idx = -1
            gs_used = guidance_scale
        else:
            routed_idx = int(source_to_teacher[source])
            gs_used = teacher_gs[routed_idx]

        with torch.no_grad():
            if in_warmup:
                student.eval()
                with student.disable_adapter():
                    teacher_cond = velocity(
                        student, z_tau, tau * 1000, prompt_embeds, pooled
                    )
                    teacher_uncond = velocity(
                        student, z_tau, tau * 1000, neg_embeds, neg_pooled
                    )
                student.train()
            else:
                teacher = teachers[routed_idx]
                teacher_cond = velocity(teacher, z_tau, tau * 1000, prompt_embeds, pooled)
                teacher_uncond = velocity(
                    teacher, z_tau, tau * 1000, neg_embeds, neg_pooled
                )
            real_cfg = teacher_uncond + gs_used * (teacher_cond - teacher_uncond)
            critic.eval()
            v_critic_eval = velocity(critic, z_tau, tau * 1000, prompt_embeds, pooled)
            delta = real_cfg - v_critic_eval
            total_grad = expand_time(tau) * delta

        target_z = (z_end - total_grad).detach()
        loss_student = pseudo_huber_loss(z_end, target_z, huber_c)
        loss_dmd_value = float(loss_student.detach().item())

        loss_vmse_value = 0.0
        omega_stats: Optional[Dict[str, float]] = None
        tau_mean = float(tau.detach().float().mean().item())
        vmse_active = (
            vmse_weight > 0.0
            and global_step >= vmse_start
            and tau_mean <= tau_threshold
        )
        if vmse_active:
            v_student_tau = velocity(student, z_tau, tau * 1000, prompt_embeds, pooled)
            squared = (real_cfg - v_student_tau) ** 2
            if use_wb:
                omega, omega_raw = wb_omega(
                    tau,
                    sigma_upper,
                    sigma_lower,
                    mode=wb_mode,
                    eps=wb_eps,
                    normalize=wb_normalize,
                )
                loss_vmse = (squared * expand_time(omega.to(squared.dtype))).mean()
                omega_f = omega.detach().float()
                raw_f = omega_raw.detach().float()
                omega_stats = {
                    "omega_min": float(omega_f.min().item()),
                    "omega_mean": float(omega_f.mean().item()),
                    "omega_max": float(omega_f.max().item()),
                    "omega_median": float(omega_f.median().item()),
                    "omega_raw_mean": float(raw_f.mean().item()),
                    "omega_raw_max": float(raw_f.max().item()),
                }
            else:
                loss_vmse = squared.mean()
            loss_vmse_value = float(loss_vmse.detach().item())
            loss_student = loss_student + vmse_weight * loss_vmse

        loss_student.backward()
        allreduce_mean_grads(student_params, world_size)
        student_grad_norm = torch.nn.utils.clip_grad_norm_(
            student_params, float(cfg["clip_grad_norm"])
        )
        if torch.isfinite(student_grad_norm):
            student_opt.step()
            student_sched.step()
        student_opt.zero_grad(set_to_none=True)

        record = {
            "step": global_step,
            "source": source,
            "routed_teacher": (
                "warmup_frozen_student" if routed_idx < 0 else teacher_names[routed_idx]
            ),
            "teacher_gs": gs_used,
            "loss": float(loss_student.detach().item()),
            "loss_dmd": loss_dmd_value,
            "loss_critic": float(loss_critic.detach().item()),
            "loss_lownoise_vmse": loss_vmse_value,
            "selected_segment": selected,
            "sigma_upper": sigma_upper,
            "sigma_lower": sigma_lower,
            "tau_mean": tau_mean,
            "is_weight_mean": float(is_weight.detach().float().mean().item()),
            "grad_norm": float(student_grad_norm.item()),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "lr": student_opt.param_groups[0]["lr"],
            "critic_lr": critic_opt.param_groups[0]["lr"],
            "step_time_s": round(time.time() - t_start, 2),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        if omega_stats is not None:
            record.update(omega_stats)
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        t_start = time.time()

        if rank == 0 and global_step % int(cfg["log_interval"]) == 0:
            omega_text = ""
            if omega_stats is not None:
                omega_text = (
                    f" wb[min={omega_stats['omega_min']:.4f}"
                    f" mean={omega_stats['omega_mean']:.6f}"
                    f" max={omega_stats['omega_max']:.4f}"
                    f" raw_mean={omega_stats['omega_raw_mean']:.4f}]"
                )
            print(
                f"[step {global_step:>6}] src={source:<9}"
                f" tea={record['routed_teacher']:<22}"
                f" loss={record['loss']:.5f} dmd={loss_dmd_value:.5f}"
                f" critic={record['loss_critic']:.5f}"
                f" vmse={loss_vmse_value:.5f}"
                f" seg={selected} tau={tau_mean:.4f}"
                f" gn={record['grad_norm']:.3f} lr={record['lr']:.2e}"
                f" {record['step_time_s']}s mem={record['peak_mem_gb']}G"
                + omega_text,
                flush=True,
            )

        save_interval = int(cfg["save_interval"])
        is_last = global_step + 1 == max_steps
        if save_interval > 0 and (
            (global_step > 0 and global_step % save_interval == 0) or is_last
        ):
            target = os.path.join(ckpt_dir, f"checkpoint-step-{global_step}")
            marker = complete_marker_path(target)
            if rank == 0:
                try:
                    os.unlink(marker)
                except FileNotFoundError:
                    pass
                save_adapter(student, os.path.join(target, "transformer"))
                save_adapter(
                    critic, os.path.join(target, "extra_models", "critic", "transformer")
                )
                os.makedirs(os.path.join(target, TRAIN_STATE_SUBDIR), exist_ok=True)
            if dist.is_initialized():
                dist.barrier()
            save_id = [uuid.uuid4().hex if rank == 0 else None]
            if dist.is_initialized():
                dist.broadcast_object_list(save_id, src=0)
            save_training_state(
                target,
                rank,
                world_size,
                global_step,
                device,
                student,
                critic,
                student_opt,
                critic_opt,
                student_sched,
                critic_sched,
                checkpoint_id=save_id[0],
                config_fingerprint=config_fingerprint,
            )
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                with open(f"{marker}.tmp", "w", encoding="ascii") as handle:
                    handle.write(save_id[0])
                os.replace(f"{marker}.tmp", marker)
                print(f"[save] {target}", flush=True)

    metrics_file.close()
    log0("[done] training finished")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

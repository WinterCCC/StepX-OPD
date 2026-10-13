"""Sigma schedules and the forward (re-noising) law for rectified-flow SD3.

Three things live here:

* ``calculate_shift`` / ``retrieve_timesteps`` are the standard diffusers
  helpers, vendored so that the training grid and the four-step sampling grid
  are built by exactly the same code path as the reference implementation.
* ``build_stratified_random_grid`` is the randomised student grid: the
  monotone sigma nodes are jittered inside disjoint strata of the PRE-shift
  uniform space, then mapped through the scheduler's shift transform.
* ``flow_matching_renoise_sde`` is the exact forward law used to move a
  trajectory point from one sigma to a larger one.

The rectified-flow convention throughout is ``x_t = (1 - t) * x_0 + t * eps``,
so ``t = sigma``, ``sigma = 1`` is pure noise and ``sigma = 0`` is clean.
"""

from __future__ import annotations

import inspect
import random
from typing import List, Optional, Tuple, Union

import torch


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Resolution-dependent shift exponent mu (diffusers' flux/sd3 formula)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Call ``scheduler.set_timesteps`` and return ``(timesteps, n_steps)``.

    Vendored from diffusers so that passing custom ``sigmas`` and ``mu``
    behaves identically to the reference sampler.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts:
            raise ValueError(
                f"{scheduler.__class__} does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts:
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def effective_shift(scheduler, mu: Optional[float]) -> float:
    """The shift factor the scheduler will actually apply.

    Under dynamic shifting the factor is ``exp(mu)``; otherwise it is the
    static ``shift`` from the scheduler config. SD3.5-Medium ships without
    ``use_dynamic_shifting``, so this returns 3.0 there.
    """
    import math

    if scheduler.config.get("use_dynamic_shifting", False) and mu is not None:
        return math.exp(float(mu))
    shift = getattr(scheduler, "shift", None)
    if shift is None:
        shift = scheduler.config.get("shift", 3.0)
    return float(shift)


def stratified_random_u_nodes(num_steps: int, seed: int) -> List[float]:
    """PRE-shift stratified-random nodes for the student grid.

    The deterministic grid is a linspace in this pre-shift uniform space. Here
    ``[0, 1]`` is cut into ``num_steps`` equal strata, the top node is pinned to
    ``u = 1`` so that ``sigma = 1`` survives the shift, and every remaining
    stratum contributes one uniform draw. Because the strata are disjoint and
    traversed in order, the result is strictly decreasing by construction and
    each node stays inside the stratum that holds the corresponding
    deterministic node, which makes this a stratified jitter of the baseline
    rather than a different schedule.

    The draw depends only on ``seed``, so every data-parallel rank that passes
    the same training step rebuilds the identical grid without a collective.
    """
    num_steps = int(num_steps)
    if num_steps < 2:
        raise ValueError("stratified_random_u_nodes requires num_steps >= 2")
    rng = random.Random(int(seed))
    u_nodes = [1.0]
    for i in range(1, num_steps):
        hi = (num_steps - i) / num_steps
        lo = (num_steps - i - 1) / num_steps
        u_nodes.append(lo + (hi - lo) * rng.random())
    return u_nodes


GRID_SEED_OFFSET = 77901


def build_stratified_random_grid(
    num_steps: int,
    global_step: int,
    shift_factor: float,
    device,
    num_train_timesteps: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomised student grid for one training step.

    Returns ``(timesteps[num_steps], sigmas[num_steps + 1])`` with
    ``timesteps = sigmas[:-1] * num_train_timesteps``, mirroring what the
    scheduler would hand back for a deterministic grid. The pre-shift nodes go
    through the same ``sigma = s*u / (1 + (s-1)*u)`` transform the scheduler
    applies, and the exact terminal ``sigma = 0`` is appended, so both endpoints
    of the deterministic grid are preserved.
    """
    u_nodes = stratified_random_u_nodes(num_steps, int(global_step) + GRID_SEED_OFFSET)
    s = float(shift_factor)
    sigma_vals = [s * u / (1.0 + (s - 1.0) * u) for u in u_nodes]
    sigma_vals.append(0.0)
    sigmas = torch.tensor(sigma_vals, device=device, dtype=torch.float32)
    timesteps = sigmas[:-1] * float(num_train_timesteps)
    return timesteps, sigmas


def flow_matching_renoise_sde(
    x_curr: torch.Tensor,
    t_curr: torch.Tensor,
    t_target: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Move a trajectory point from ``t_curr`` to a noisier ``t_target``.

    Both time tensors broadcast against ``x_curr`` (shape ``[B, 1, 1, 1]`` for
    image latents). The map is the variance-correct forward law of the linear
    path: write ``alpha = 1 - t`` and ``sigma = t``, then
    ``x_target = (alpha_target / alpha_curr) * x_curr + beta * noise`` with
    ``beta^2 = sigma_target^2 - (scale * sigma_curr)^2``. At ``t_curr = 0`` it
    reduces to the plain forward process from a clean sample.
    """
    alpha_curr = torch.clamp(1.0 - t_curr, min=1e-5)
    alpha_target = 1.0 - t_target
    scale = alpha_target / alpha_curr
    var_term = torch.clamp(t_target**2 - (scale * t_curr) ** 2, min=0.0)
    beta = torch.sqrt(var_term)
    return scale * x_curr + beta * noise

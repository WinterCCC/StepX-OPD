#!/usr/bin/env python3
"""Numeric self-check of the w_b weight, with no model or GPU involved.

Three things are printed. The first reproduces the draw that a training step
performs, so the min, mean and max of the normalised weight can be read off: the
mean is one by construction of the normalisation, so a mean that is not one
means the normalisation is not reaching the loss. The second sweeps the epsilon
in the denominator and reports the largest weight a single sample can reach,
which is what motivates the chosen value. The third reports the same statistics
for the two alternative forms of the interval term.

Run: ``python tools/wb_weight_selfcheck.py``
"""

from __future__ import annotations

import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stepx_opd.objective import wb_omega
from stepx_opd.schedule import build_stratified_random_grid

SEED_SELECT_SEGMENT = 11
SEED_TAU = 12345


def draw_step(global_step: int, num_steps: int, bsz: int, shift: float = 3.0):
    """Reproduce one training step's grid, segment choice and tau vector."""
    _, sigmas = build_stratified_random_grid(num_steps, global_step, shift, "cpu")
    selected = random.Random(global_step + SEED_SELECT_SEGMENT).randrange(num_steps)
    sigma_upper = float(sigmas[selected].item())
    sigma_lower = float(sigmas[selected + 1].item())
    gen = torch.Generator().manual_seed(global_step + SEED_TAU)
    span = max(sigma_upper - sigma_lower, 1e-6)
    tau_base = sigma_lower + torch.rand((bsz,), generator=gen) * span
    u_tau = torch.rand((bsz,), generator=gen)
    tau = tau_base + u_tau * (sigma_upper - tau_base).clamp(min=0.0)
    return tau, sigma_upper, sigma_lower, selected


def main() -> None:
    num_steps, bsz = 4, 64
    print("=" * 74)
    print(" normalised w_b over the steps whose tau falls in the low-noise half")
    print("=" * 74)
    shown = 0
    for step in range(500, 900):
        tau, su, sl, selected = draw_step(step, num_steps, bsz)
        if float(tau.mean()) > 0.5:
            continue
        omega, raw = wb_omega(tau, su, sl, mode="upper", eps=0.05, normalize=True)
        print(
            f" step={step:>4} seg={selected} sigma=[{sl:.4f},{su:.4f}]"
            f" tau_mean={float(tau.mean()):.4f} |"
            f" min={float(omega.min()):.4f}"
            f" mean={float(omega.mean()):.6f}"
            f" median={float(omega.median()):.4f}"
            f" max={float(omega.max()):.4f} |"
            f" raw_mean={float(raw.mean()):.4f} raw_max={float(raw.max()):.4f}"
        )
        shown += 1
        if shown >= 12:
            break

    print()
    print("=" * 74)
    print(" largest single-sample weight as a function of the denominator epsilon")
    print("=" * 74)
    for eps in (0.0, 0.001, 0.01, 0.05, 0.1):
        worst_raw = 0.0
        worst_norm = 0.0
        for step in range(500, 2000):
            tau, su, sl, _ = draw_step(step, num_steps, bsz)
            if float(tau.mean()) > 0.5:
                continue
            omega, raw = wb_omega(tau, su, sl, mode="upper", eps=eps, normalize=True)
            worst_raw = max(worst_raw, float(raw.max()))
            worst_norm = max(worst_norm, float(omega.max()))
        print(f" eps={eps:<6} raw_max={worst_raw:10.3f}  normalised_max={worst_norm:8.3f}")

    print()
    print("=" * 74)
    print(" the three forms of the interval term, over the same gated steps")
    print("=" * 74)
    for mode in ("upper", "lower", "seg"):
        maxima = []
        for step in range(500, 2000):
            tau, su, sl, _ = draw_step(step, num_steps, bsz)
            if float(tau.mean()) > 0.5:
                continue
            omega, _ = wb_omega(tau, su, sl, mode=mode, eps=0.05, normalize=True)
            maxima.append(float(omega.max()))
        maxima_t = torch.tensor(maxima)
        print(
            f" mode={mode:<6} normalised max over steps:"
            f" median={float(maxima_t.median()):.3f}"
            f" p99={float(maxima_t.quantile(0.99)):.3f}"
            f" max={float(maxima_t.max()):.3f}"
            f"  (n_gated_steps={len(maxima)})"
        )


if __name__ == "__main__":
    main()

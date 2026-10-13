"""Loss terms of the four-step distribution-matching objective.

Four pieces are defined here and nothing else: the 2x2 patch grouping that the
pseudo-Huber surrogate is computed over, the surrogate itself, the importance
weight that corrects the critic regression for the actual re-noising source,
and the ``w_b`` weight that this variant puts on the low-noise velocity
regression.
"""

from __future__ import annotations

from typing import Tuple

import torch


def pack_2x2(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W] -> [B, (H/2)*(W/2), C*4]``, SD3's native patch layout."""
    b, c, h, w = latents.shape
    x = latents.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_2x2(latents: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
    """``[B, L, C*4] -> [B, C, lat_h, lat_w]``, inverse of :func:`pack_2x2`."""
    b, _, cc = latents.shape
    c = cc // 4
    x = latents.view(b, lat_h // 2, lat_w // 2, c, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, lat_h, lat_w)


def pseudo_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    c_coef: float = 0.00054,
) -> torch.Tensor:
    """Pseudo-Huber distance between two latents, grouped per 2x2 patch.

    The squared error is summed over one patch token (``C * 4`` channels), the
    constant ``c = c_coef * sqrt(d)`` with ``d`` the token dimension keeps the
    surrogate scale-aware, and the mean runs over tokens and batch. Grouping at
    the patch level is what the reference implementation does, so the constant
    is calibrated for it and a per-element variant would not be comparable.
    """
    pred_p = pack_2x2(pred)
    target_p = pack_2x2(target)
    d = pred_p.shape[-1] * pred_p.shape[-2]
    c = float(c_coef) * (d**0.5)
    diff = pred_p - target_p
    return (
        torch.sqrt(diff.pow(2).sum(dim=-1, keepdim=True) + c * c) - c
    ).mean()


def conditional_is_weight(
    x_target: torch.Tensor,
    x_start: torch.Tensor,
    x0_pred: torch.Tensor,
    t_start: torch.Tensor,
    t_target: torch.Tensor,
) -> torch.Tensor:
    """Per-sample density ratio ``q(x_t | x0_hat) / q(x_t | x_s)``.

    The denominator is the proposal that actually produced ``x_target``, namely
    the re-noising of ``x_start`` at ``t_start`` performed by
    :func:`stepx_opd.schedule.flow_matching_renoise_sde`. The numerator is the
    plain forward process from the corresponding clean estimate. Both log
    densities are averaged over a sample's latent dimensions before
    exponentiation, which is what keeps the weight in a usable range, and the
    exponent is clamped to ``[-20, 20]``.

    ``t_start`` and ``t_target`` are ``(B,)``; the returned weight is ``(B,)``.
    """
    bsz = x_target.shape[0]
    xt = x_target.reshape(bsz, -1).float()
    xs = x_start.reshape(bsz, -1).float()
    x0 = x0_pred.reshape(bsz, -1).float()

    tau = t_target.reshape(bsz, 1).float().clamp(min=1e-5, max=1.0)
    sigma_start = t_start.reshape(bsz, 1).float().clamp(min=0.0, max=1.0 - 1e-5)
    alpha_start = (1.0 - sigma_start).clamp(min=1e-5)
    alpha_target = (1.0 - tau).clamp(min=0.0)
    scale = alpha_target / alpha_start
    beta = (tau.square() - (scale * sigma_start).square()).clamp(min=1e-10).sqrt()

    residual_num = (xt - alpha_target * x0) / tau
    residual_den = (xt - scale * xs) / beta
    log_q_num = -0.5 * (residual_num.square() + 2.0 * torch.log(tau))
    log_q_den = -0.5 * (residual_den.square() + 2.0 * torch.log(beta))
    log_weight = (log_q_num - log_q_den).mean(dim=1).clamp(min=-20.0, max=20.0)
    return torch.exp(log_weight)


def critic_omega(tau: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """``(1 - tau) / (tau + eps)``, the weight on the critic regression."""
    return (1.0 - tau) / (tau + eps)


def wb_omega(
    tau: torch.Tensor,
    sigma_upper: float,
    sigma_lower: float,
    mode: str = "upper",
    eps: float = 0.05,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample weight for the low-noise velocity regression.

    The factor is ``w(tau) = delta * (1 - tau) / (tau + eps)``. It comes from
    the marginal-KL decomposition ``KL(q_b || p_b) = D_b + E[w_b(T) * ||v_q -
    v_p||^2]``, where the factor multiplying the squared velocity difference is
    ``(t_i - t)(1 - t) / t`` with ``t_i`` the upper end of the interval that
    ``tau`` was drawn from. The default ``mode="upper"`` therefore takes
    ``delta = sigma_upper - tau``, which is the form the derivation gives;
    ``"lower"`` (``tau - sigma_lower``) and ``"seg"`` (``sigma_upper -
    sigma_lower``) are kept switchable for inspection. This is a different
    quantity from the conditional-transition weight of a single SDE step: that
    one needs a diffusion coefficient, and the student rollout here is a
    deterministic Euler solve.

    Two conventions matter when reading the result. First, the denominator is
    ``tau + eps`` rather than ``tau``: in the terminal interval ``sigma_lower``
    is exactly zero, so ``tau`` can reach ``1e-5`` and an unregularised ``1 /
    tau`` lets a single sample take over the batch mean; ``eps = 0.05`` caps the
    largest value near 8 while leaving roughly a sixfold spread. Second, with
    ``normalize=True`` the weight is divided by its own batch mean, so
    ``E[w] = 1`` and what enters the loss is the relative weighting between
    samples at an unchanged overall scale. That is why the term's own
    coefficient stays at the unweighted value.

    Returns ``(normalized, raw)``, both ``(B,)``. The draw of ``tau`` depends
    only on the training step, so the batch mean is identical on every rank and
    the normalisation needs no collective.
    """
    t = tau.detach().float().reshape(-1)

    def _as_vec(value):
        if torch.is_tensor(value):
            return value.detach().float().reshape(-1).to(t.device)
        return torch.full_like(t, float(value))

    su = _as_vec(sigma_upper)
    sl = _as_vec(sigma_lower)
    if mode == "upper":
        delta = su - t
    elif mode == "lower":
        delta = t - sl
    elif mode == "seg":
        delta = su - sl
    else:
        raise ValueError(f"mode must be 'upper', 'lower' or 'seg', got {mode!r}")

    omega_raw = (
        delta.clamp(min=0.0)
        * (1.0 - t).clamp(min=0.0)
        / (t + float(eps)).clamp(min=1e-8)
    )
    omega = omega_raw
    if normalize:
        omega = omega_raw / omega_raw.mean().clamp(min=1e-8)
    return omega, omega_raw

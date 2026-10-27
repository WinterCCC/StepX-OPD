# StepX-OPD: Rethinking Multi-Teacher On-Policy Distillation in Diffusion Models

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://winterccc.github.io/Step-OPD/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-StepX--OPD--SD3.5-yellow)](https://huggingface.co/ShimaRinC/StepX-OPD-SD3.5)
![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey)

![StepX-OPD training cost and four-step benchmark coverage](assets/teaser_gpu_hours_radar_300dpi.png)

*Training cost of each optimization stage, alongside four-step performance across the evaluated benchmarks.*

This repository trains a four-step SD3.5-Medium sampler by distribution matching against a set of frozen expert teachers, and it generates images from the result.

## Abstract

StepX-OPD trains a four-step sampler for SD3.5-Medium by combining capability integration and step distillation into a single training stage. Each training mini-batch is drawn from one of several sources, and the student is routed to the frozen specialist teacher associated with that source. Supervision is attached to states sampled from the student's own short rollout rather than to a fixed timestep grid, so the distribution-matching signal targets the states the student actually visits at inference time. The student's velocity is pushed toward the routed teacher's guided velocity through a trainable critic, and in the low-noise half of the rollout an additional direct velocity regression against the teacher is applied, weighted by a per-sample term taken from a marginal-KL decomposition of the matching objective. This repository covers that weighted variant, `w_b`, and is derived from the DiffusionOPD codebase, whose SD3 multi-teacher training script supplied the prompt datasets, the prompt-encoding path, and the low-rank adapter assembly.

![Qualitative comparison across six models](assets/qualitative_comparison_sixmodel.png)

*Qualitative comparison across six models, including StepX-OPD.*

## 🚀 Getting Started

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare weights and data

| Asset | Link |
|---|---|
| SD3.5-Medium base | https://huggingface.co/stabilityai/stable-diffusion-3.5-medium |
| StepX-OPD LoRA weights | https://huggingface.co/ShimaRinC/StepX-OPD-SD3.5 |
| Three expert teacher adapters | https://huggingface.co/quanhaol/DiffusionOPD (published with the DiffusionOPD release) |

These are the `PLACEHOLDER` entries to fill in `configs/sd35m_wb.yaml`; the base model entry takes a local directory, not a hub id.

### 3. Train

```bash
GPUS=0,1,2,3 CONFIG=configs/sd35m_wb.yaml bash scripts/launch_train_wb.sh
```

`OVERRIDE` takes inline YAML merged over the config file, which is how to shorten a run or shrink the batch for a smoke test:

```bash
OVERRIDE='{max_train_steps: 40, train_batch_size: 4}' bash scripts/launch_train_wb.sh
```

### 4. Generate

```bash
python scripts/generate_sd35_4step.py \
    --base /path/to/stable-diffusion-3.5-medium \
    --init_merge /path/to/sft_merge_additive \
    --adapter /path/to/checkpoint-step-<N> \
    --prompts prompts.json \
    --out_dir out/ \
    --seeds 0 --steps 4 --cfg 1.0 --size 512
```

This script uses the sampling grid the model is evaluated with. Substituting another sampling pipeline's default grid lowers the benchmark scores, most visibly on image quality.

## 🔁 Resuming a Run

```bash
GPUS=0,1,2,3 \
RESUME_FROM=/path/to/run/checkpoint/checkpoint-step-<N> \
    bash scripts/launch_train_wb.sh
```

The launch must use the same number of ranks and the same config that wrote the checkpoint, except for `output_dir`, `run_name`, `log_interval`, `save_interval` and `max_train_steps`. A checkpoint whose save did not finish is rejected.

## 🔗 Related Links

The project page is at https://winterccc.github.io/Step-OPD/, and the trained weights are published at https://huggingface.co/ShimaRinC/StepX-OPD-SD3.5.

## 📄 License

Apache License 2.0, see `LICENSE`. The code is derived from the DiffusionOPD codebase, which carries the same license, and `NOTICE` records that attribution. Weights produced with this code apply on top of Stable Diffusion 3.5 Medium and are subject to the Stability AI Community License that governs that base model.

## 📚 Citation

The paper this code accompanies has not been published yet. Citation information will be added here once it is released.

## 📭 Contact

For questions or issues, please open a GitHub issue in this repository.

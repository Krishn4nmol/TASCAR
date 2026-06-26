# TWASAC: Temporal Workload-Aware Cold Start Mitigation in Serverless Computing via Transformer-Attention Soft Actor-Critic

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-IEEE%20Access%20Submitted-brightgreen)
![Seeds](https://img.shields.io/badge/Seeds-7%20Independent-purple)
![Reduction](https://img.shields.io/badge/CSR%20Reduction-11.97--18.57pp-red)
![Metrics](https://img.shields.io/badge/Metrics-18%20Comprehensive-blue)

> **Note:** This repository was originally named TASCAR (working title). The code implements the TWASAC architecture described in the paper. Result folders and model folders retain the TASCAR name as they are referenced by evaluation scripts.

---

## Overview

**TWASAC** is a novel serverless container scheduling framework that significantly outperforms the **CASR** baseline (Chen et al., *Future Gener. Comput. Syst.*, 2025).

TWASAC replaces CASR's PPO agent with three architectural innovations:

- **Transformer encoder** — processes sequences of 10 historical cache states with cross-queue attention, learning temporal workload patterns invisible to single-snapshot approaches
- **Soft Actor-Critic (SAC)** — off-policy learning with dual critics and entropy-driven exploration, replacing PPO's on-policy updates
- **Dynamic reward weighting** — automatically adapts θ (0.5–0.9) based on observed cold start rate, replacing CASR's fixed θ=0.8

> **Key result:** TWASAC reduces cold start rate by **11.97 to 18.57 percentage points** over the strongest zero-waste-memory baseline across **7 independent random seeds** (42, 123, 456, 789, 1000, 2024, 2025), with near-zero cross-seed variance of 0.09–0.24 pp (p < 0.001, Cohen's d > 3.0).

---

## Results

### Primary Cold Start Rate Comparison (Seed 42)

| Workload | Baseline CSR | TWASAC CSR | Improvement |
|----------|-------------|------------|-------------|
| Common | 89.840% | 72.111% | **−17.729 pp** |
| Significant | 92.024% | 74.973% | **−17.051 pp** |
| Random | 86.051% | 72.377% | **−13.674 pp** |

### Multi-Seed Validation (7 Seeds: 42, 123, 456, 789, 1000, 2024, 2025)

| Workload | Baseline (mean ± std) | TWASAC (mean ± std) | Diff (pp) | p-value |
|----------|-----------------------|---------------------|-----------|---------|
| Common | 89.58 ± 1.36% | 71.59 ± 0.24% | 17.99 ± 1.28 | < 0.001 |
| Significant | 93.41 ± 1.28% | 74.84 ± 0.24% | 18.57 ± 1.25 | < 0.001 |
| Random | 82.91 ± 5.28% | 70.94 ± 0.09% | 11.97 ± 5.30 | 0.001 |

- Cohen's d > 3.0 on all workloads
- Wilcoxon signed-rank p = 0.008 on all workloads
- 95% CIs: Common [16.71, 19.27], Significant [17.32, 19.82], Random [6.68, 17.26]
- **21/21** seed-workload combinations favour TWASAC

### Equal-Budget Comparison (Common Workload)

| Configuration | Episodes | CSR |
|---------------|----------|-----|
| Baseline (200 ep) | 200 | 89.105% |
| Baseline retrained (500 ep) | 500 | 87.869% |
| TWASAC | 500 | 72.101% |

TWASAC's advantage is architectural, not a training-budget effect.

### Temporal Encoder Comparison (Seeds 42, 123, 456)

| Encoder | Common | Significant | Random | Avg Gap vs TWASAC |
|---------|--------|-------------|--------|-------------------|
| TWASAC (Transformer+SAC) | ~72% | ~75% | ~71% | — |
| LSTM+SAC | ~88% | ~88% | ~80% | 12.6 pp |
| GRU+SAC | ~95% | ~94% | ~91% | 21.1 pp |

### Ablation Study (Seed 42)

| Variant | Common | Significant | Random |
|---------|--------|-------------|--------|
| V1: Baseline (PPO) | 88.796% | 95.626% | 89.199% |
| V2: SAC-Only | 78.061% | 78.206% | 80.037% |
| V3: Transformer+PPO | 91.314% | 94.253% | 70.698% |
| V4: Full TWASAC | 72.146% | 75.521% | 68.950% |

### Baseline Comparison (Seed 42)

| Workload | vs Baseline | vs FaaSCache | vs Hist |
|----------|-------------|--------------|---------|
| Common | +17.729 pp ✅ | +27.888 pp ✅ | −10.897 pp |
| Significant | +17.051 pp ✅ | +25.027 pp ✅ | −13.331 pp |
| Random | +13.674 pp ✅ | +27.623 pp ✅ | −10.996 pp |

TWASAC outperforms FaaSCache and the baseline while maintaining **zero wasted memory time** — Hist cannot achieve this (11.7–25.7s WMT).

### Comprehensive Metrics (Seed 42, 11 wins / 4 ties / 3 losses)

| Metric | Winner |
|--------|--------|
| Cold Start Rate | TWASAC ✅ |
| P95 / P99 Latency | TWASAC ✅ |
| Average Response Time | TWASAC ✅ |
| Container Utilization (+98–214%) | TWASAC ✅ |
| Resource Utilization Efficiency | TWASAC ✅ |
| SLA Violation Rate | TWASAC ✅ |
| Energy per Request | TWASAC ✅ |
| CO₂ Estimate (−10.4–16.8%) | TWASAC ✅ |
| TPI Composite Score | TWASAC ✅ |
| Wasted Memory Time | Tie (both 0.000s) ✅ |
| Throughput / Burst Handling | Tie |
| Avg Cold Start Delay | Baseline |
| Scaling Accuracy / Elasticity | Baseline |

### RL Training Summary (Seed 42)

| Metric | Value |
|--------|-------|
| Training Time | 4512.3 s (~75 min) |
| Best Reward | −0.1351 |
| Best Checkpoint | Episode 350 |
| Total Training Samples | 50,000 |
| Sample Efficiency | −0.027023 |
| Cumulative Reward | −310.00 |
| θ Range | 0.500–0.900 |

---

## Architecture

```
Azure Function Traces
        │
        ▼
  S-Cache (W-TinyLFU, K=3 queues)
        │ state vector (21-dim)
        ▼
  State History Buffer (last 10 states)
        │ sequence (10 × 21)
        ▼
  Transformer Encoder
  ├── Linear projection: 21 → 64
  ├── Positional encoding
  ├── 2 Transformer layers, 4 heads
  ├── Cross-queue attention
  └── Output: 64-dim encoded state
        │
        ▼
  SAC Agent
  ├── Actor:    64 → 128 → 128 → 27
  ├── Critic 1: 64 → 128 → 128 → 27
  └── Critic 2: 64 → 128 → 128 → 27
        │ action (0–26, 3 queues × 3 choices)
        ▼
  Dynamic Reward θ (adapts 0.5–0.9)
        │
        ▼
  S-Cache ◄──────────── apply scaling ──┘

  MetricsTracker (non-invasive wrapper)
  └── 18 metrics computed at evaluation
```

### TWASAC vs Baseline: Architecture Comparison

| Component | Baseline (CASR) | TWASAC |
|-----------|-----------------|--------|
| RL Algorithm | PPO (on-policy) | SAC (off-policy) |
| State Input | Single snapshot (21-dim) | Sequence of 10 states |
| Temporal Model | None | Transformer encoder |
| Cross-queue reasoning | Independent | Cross-queue attention |
| Reward weighting | Fixed θ=0.8 | Dynamic θ (0.5–0.9) |
| Exploration | Clipped gradient | Entropy temperature |
| Decisions/episode | 10 | 100 |
| Sample reuse | No | Yes (replay buffer 100K) |
| Critics | 1 | 2 (reduces overestimation) |
| Training seeds | 1 | 7 |

---

## Project Structure

```
TWASAC/
├── Core
│   ├── config.py                     ← All hyperparameters
│   ├── simulator.py                  ← Azure dataset loader
│   ├── scache.py                     ← W-TinyLFU S-Cache (K=3 queues)
│   ├── environment.py                ← RL environment wrapper
│   ├── transformer_encoder.py        ← Transformer + StateHistoryBuffer
│   ├── sac_agent.py                  ← SAC with dual critics
│   ├── ppo_agent.py                  ← PPO baseline agent
│   ├── metrics_tracker.py            ← 18-metric evaluation wrapper
│   └── baselines.py                  ← FaaSCache and Hist baselines
│
├── Training
│   ├── train_twasac.py               ← Main TWASAC training (was train_tascar.py)
│   ├── train_sac_only.py             ← Ablation V2
│   ├── train_transformer_ppo.py      ← Ablation V3
│   ├── train_casr200_seed123.py      ← Ablation V1 seed 123
│   ├── train_casr200_seed456.py      ← Ablation V1 seed 456
│   ├── train_casr_500.py             ← Equal-budget baseline
│   ├── train_lstm_sac.py             ← LSTM+SAC encoder
│   └── train_gru_sac.py              ← GRU+SAC encoder
│
├── Evaluation
│   ├── evaluate_twasac.py            ← Full evaluation pipeline
│   ├── evaluate.py                   ← General evaluator
│   ├── eval_generalization.py        ← Cross-workload generalization
│   ├── eval_sensitivity.py           ← Hyperparameter sensitivity
│   ├── eval_temporal_comparison.py   ← LSTM/GRU comparison
│   ├── eval_temporal_multiseed.py    ← Multi-seed temporal eval
│   ├── ablation_study.py             ← 4-variant ablation
│   ├── multiseed_ablation_eval.py    ← 12-model ablation eval
│   └── find_best_checkpoint.py       ← Checkpoint selection
│
├── Multi-Seed Runs
│   ├── run_multiseed.py              ← Seeds 42, 123, 456
│   ├── run_multiseed_part2.py        ← Seeds 789, 1000, 2024, 2025
│   ├── run_twasac_seed789.py         ← Seed 789 wrapper
│   ├── run_twasac_common_only.py     ← Common workload only
│   ├── run_lstm_sac_seed42.py        ← LSTM seeds
│   ├── run_lstm_sac_seed123.py
│   ├── run_lstm_sac_seed456.py
│   ├── run_gru_sac_seed42.py         ← GRU seeds
│   ├── run_gru_sac_seed123.py
│   ├── run_gru_sac_seed456.py
│   ├── run_sac_only_seed123.py       ← Ablation V2
│   ├── run_sac_only_seed456.py
│   ├── run_transformer_ppo_seed123.py ← Ablation V3
│   ├── run_transformer_ppo_seed456.py
│   └── run_sensitivity_*.py          ← Sensitivity analysis (6 configs)
│
├── Statistics & Figures
│   ├── run_statistical_tests.py      ← t-test, Wilcoxon, Cohen's d
│   ├── run_fixed_theta.py            ← Dynamic θ ablation
│   ├── plot_convergence_comparison.py
│   ├── regenerate_all_figures.py     ← Regenerate all paper figures
│   ├── measure_latency.py
│   └── figures_twasac/               ← All paper figures (300 DPI)
│       ├── multiseed_comparison.png
│       ├── per_seed_comparison.png
│       ├── ablation_comparison.png
│       ├── fig_convergence_comparison.png
│       ├── fig1_cold_start.png
│       ├── fig2_latency_memory.png
│       ├── fig3_resource.png
│       ├── fig4_qos_throughput.png
│       ├── fig5_energy_scaling.png
│       ├── fig6_tpi_agi.png
│       └── fig7_rl_metrics.png
│
└── Results (JSON)
    ├── results_tascar/               ← Primary seed 42 results
    ├── results_tascar_seed123/       ← Seed 123
    ├── results_tascar_seed456/       ← Seed 456
    ├── results_tascar_seed789/       ← Seed 789
    ├── results_tascar_seed1000/      ← Seed 1000
    ├── results_tascar_seed2024/      ← Seed 2024
    ├── results_tascar_seed2025/      ← Seed 2025
    ├── results_multiseed/            ← Seeds 42/123/456 combined
    ├── results_multiseed_part2/      ← Seeds 789/1000/2024/2025
    ├── results_statistical/          ← t-test, Wilcoxon, CIs
    ├── results_ablation/             ← Full ablation + sensitivity
    ├── results_fixed_theta*/         ← Dynamic θ experiments
    ├── results_lstm_sac*/            ← LSTM encoder results
    └── results_gru_sac*/             ← GRU encoder results
```

---

## Installation

```bash
git clone https://github.com/Krishn4nmol/TWASAC.git
cd TWASAC

conda create -n twasac_env python=3.11
conda activate twasac_env

pip install -r requirements.txt
```

### Requirements
- Python 3.11
- PyTorch 2.11
- NumPy, SciPy, Matplotlib
- 32GB RAM recommended
- No GPU required (CPU training)

---

## Dataset

Download the [Microsoft Azure Functions 2019 Dataset](https://github.com/Azure/AzurePublicDataset):

```bash
mkdir data
# Download invocations_per_function_md.anon.d01.csv through d07.csv
# Place in data/ folder
```

**Workload definitions:**
- **Common** — top 2,000 most frequent functions, Day 1, 100K calls
- **Significant** — top 2,000 high cold-start-overhead functions, Day 2, 100K calls
- **Random** — 2,000 randomly selected functions, Day 3, 100K calls

---

## How to Run

### Train TWASAC (seed 42)
```bash
python train_twasac.py
# Takes ~75 min. Checkpoints every 50 episodes.
```

### Full 7-seed validation
```bash
python run_multiseed.py         # Seeds 42, 123, 456
python run_multiseed_part2.py   # Seeds 789, 1000, 2024, 2025
python run_statistical_tests.py # p-values, CIs, Cohen's d
```

### Full evaluation (all metrics)
```bash
python evaluate_twasac.py
# Generates all figures and metrics JSON
```

### Ablation study (12 models)
```bash
python train_casr200_seed123.py
python train_casr200_seed456.py
python run_sac_only_seed123.py
python run_sac_only_seed456.py
python run_transformer_ppo_seed123.py
python run_transformer_ppo_seed456.py
python multiseed_ablation_eval.py
```

### Temporal encoder comparison
```bash
python run_lstm_sac_seed42.py
python run_lstm_sac_seed123.py
python run_lstm_sac_seed456.py
python run_gru_sac_seed42.py
python run_gru_sac_seed123.py
python run_gru_sac_seed456.py
python eval_temporal_multiseed.py
```

### Regenerate all paper figures
```bash
python regenerate_all_figures.py
# Saves to figures_twasac/ at 300 DPI
```

---

## Hyperparameters

| Parameter | Baseline (CASR) | TWASAC |
|-----------|-----------------|--------|
| Episodes | 200 | 500 |
| Steps/episode | 10 | 100 |
| Learning rate | 0.001 | 0.0001 |
| Replay buffer | — | 100,000 |
| Batch size | 20 | 64 |
| SAC updates/step | — | 10 |
| Sequence length | 1 | 10 |
| Transformer dim | — | 64 |
| Attention heads | — | 4 |
| Transformer layers | — | 2 |
| Discount factor γ | 0.63 | 0.63 |
| θ | 0.8 (fixed) | 0.5–0.9 (dynamic) |
| Training seeds | 42 | 42,123,456,789,1000,2024,2025 |

---

## Metrics Reference

| Category | Metric | Better |
|----------|--------|--------|
| Cold Start | CSR, ACSD, P95, P99 | Lower |
| Resource | CUR, RUE | Higher |
| Resource | WMT | Lower |
| QoS | ART, SVR | Lower |
| Throughput | TPT, SER, BHE | Higher |
| Energy | EPR, CO₂ | Lower |
| Scalability | SA, ES | Higher |
| Composite | TPI, AGI | Higher |

**TPI formula:**
```
TPI = 0.25×(1−CSR) + 0.20×(1−WMT_n) + 0.20×TPT_n + 0.20×(1−SVR) + 0.15×RUE
```

---

## Citation

```bibtex
@article{gupta2026twasac,
  title={{TWASAC}: Temporal Workload-Aware Cold Start
         Mitigation in Serverless Computing via
         Transformer-Attention Soft Actor-Critic},
  author={Gupta, Udit and Krishna, Anmol
          and Misra, Rajiv},
  journal={IEEE Access},
  year={2026},
  doi={10.1109/ACCESS.2026.0000000}
}
```

---

## References

[1] Y. Chen et al., "CASR: Optimizing cold start and resource utilization
in serverless computing," *Future Gener. Comput. Syst.*, vol. 170, 2025.

[2] M. Shahrad et al., "Serverless in the Wild," *USENIX ATC*, 2020.

[3] T. Haarnoja et al., "Soft Actor-Critic," *ICML*, 2018.

[4] A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.

[5] A. Fuerst and P. Sharma, "FaaSCache," *ASPLOS*, 2021.

---

## Acknowledgments

The authors thank Microsoft Azure for making the Azure Functions 2019
dataset publicly available. This work builds on the CASR framework by
Chen et al. (2025).
---

## License

MIT License — see [LICENSE](LICENSE) for details.

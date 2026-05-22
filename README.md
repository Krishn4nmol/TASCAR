# TASCAR: Transformer-Attention Soft Actor-Critic for Adaptive Resource Optimization in Serverless Computing

## Adaptive Resource Optimization using Temporal Attention and SAC



![Python](https://img.shields.io/badge/Python-3.11-blue)




![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)




![License](https://img.shields.io/badge/License-MIT-green)




![Status](https://img.shields.io/badge/Status-Complete-brightgreen)




![Beats CASR](https://img.shields.io/badge/Beats%20CASR-16--21%20pp-red)



---

## Overview

This repository presents **TASCAR**, a novel serverless container
scheduling system that extends and significantly outperforms
**CASR** (Chen et al., Future Generation Computer Systems, 2025).

TASCAR replaces CASR's PPO reinforcement learning agent with:
- **Transformer encoder** for temporal workload modeling
- **Soft Actor-Critic (SAC)** for better exploration
- **Dynamic reward adaptation** instead of fixed theta

> **Result:** TASCAR reduces cold start rate by **16 to 21
> percentage points** compared to CASR while maintaining
> **zero wasted memory time** across all workloads!

---

## Key Results

### TASCAR vs CASR (Main Comparison)

| Workload | CASR Cold% | TASCAR Cold% | Improvement | WMT |
|----------|-----------|--------------|-------------|-----|
| Common | 89.178% | 71.493% | **✅ -17.685 pp** | 0.000s |
| Significant | 91.449% | 75.051% | **✅ -16.398 pp** | 0.000s |
| Random | 85.138% | 69.026% | **✅ -16.112 pp** | 0.000s |

### Key Findings
- TASCAR reduces cold start rate by **16.1 to 17.7 percentage points**
  over CASR across all workload types
- Both TASCAR and CASR maintain **zero wasted memory time** consistently
- TASCAR also reduces cold start **overhead** by 1-2 seconds
- Dynamic theta adapts from **0.5 to 0.9** based on workload

---

## TASCAR Architecture

```
┌─────────────────────────────────────────────────────┐
│                  TASCAR System                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Azure Function Traces                               │
│         │                                            │
│         ▼                                            │
│  ┌─────────────┐                                     │
│  │  S-Cache    │ ← W-TinyLFU K=3 queues              │
│  │  (K=3)      │   Queue 0: 0-1s   (9.4%)            │
│  └──────┬──────┘   Queue 1: 1-60s  (85.3%)           │
│         │           Queue 2: 60+s   (5.0%)            │
│         │ state (21 numbers)                          │
│         ▼                                            │
│  ┌─────────────────────────────┐                     │
│  │   State History Buffer      │                     │
│  │   Last 10 states stored     │                     │
│  └──────────────┬──────────────┘                     │
│                 │ sequence (10×21 = 210 numbers)      │
│                 ▼                                    │
│  ┌─────────────────────────────┐                     │
│  │   Transformer Encoder       │                     │
│  │   ┌─────────────────────┐   │                     │
│  │   │ Positional Encoding │   │                     │
│  │   └──────────┬──────────┘   │                     │
│  │              │               │                     │
│  │   ┌──────────▼──────────┐   │                     │
│  │   │ Transformer Layers  │   │                     │
│  │   │ (2 layers, 4 heads) │   │                     │
│  │   └──────────┬──────────┘   │                     │
│  │              │               │                     │
│  │   ┌──────────▼──────────┐   │                     │
│  │   │ Cross-Queue Attn    │   │ ← Queue interaction │
│  │   └──────────┬──────────┘   │                     │
│  └─────────────────────────────┘                     │
│                 │ enriched state (64 numbers)         │
│                 ▼                                    │
│  ┌─────────────────────────────┐                     │
│  │      SAC Agent              │                     │
│  │   ┌─────────┐ ┌──────────┐  │                     │
│  │   │  Actor  │ │ Critic×2 │  │                     │
│  │   └────┬────┘ └──────────┘  │                     │
│  │        │ entropy exploration │                     │
│  └────────┼────────────────────┘                     │
│           │ action (0-26)                            │
│           ▼                                          │
│  ┌─────────────────────────────┐                     │
│  │   Dynamic Reward Module     │                     │
│  │   θ adapts: 0.5 → 0.9      │                     │
│  └─────────────────────────────┘                     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## TASCAR vs CASR: Architecture Comparison

| Component | CASR | TASCAR |
|-----------|------|--------|
| RL Algorithm | PPO (on-policy) | SAC (off-policy) |
| State Input | Single snapshot (21 dim) | Sequence of 10 states |
| Temporal Model | None | Transformer encoder |
| Cross-queue | Independent | Cross-queue attention |
| Reward | Fixed theta=0.8 | Dynamic theta (0.5-0.9) |
| Exploration | Clipped gradient | Entropy temperature |
| Training Steps | 10 per episode | 100 per episode |
| Sample Reuse | No (on-policy) | Yes (replay buffer) |
| Critics | 1 | 2 (reduces bias) |

---

## Why TASCAR Beats CASR

### 1. Temporal State Modeling
```
CASR sees:
Current state only
[cap, len, invocations, cold, evict, running, wmt]
× 3 queues = 21 numbers

TASCAR sees:
Last 10 states as sequence!
21 × 10 = 210 numbers → Transformer → 64 enriched numbers

TASCAR learns:
- Burst patterns (sudden traffic spikes)
- Periodic patterns (daily cycles)
- Long range dependencies
- Cross-queue relationships
```

### 2. SAC vs PPO
```
PPO (CASR):
On-policy: throws away old experiences
10 decisions per episode
Gets stuck in local optima
Fixed exploration rate

SAC (TASCAR):
Off-policy: reuses ALL past experiences
100 decisions per episode
Entropy-driven exploration
Automatic adaptation
```

### 3. Dynamic Theta
```
CASR: theta = 0.8 always
Fixed balance between cold starts and memory

TASCAR: theta adapts automatically!
Cold% > 95%: increase theta → focus on cold starts!
Cold% < 85%: decrease theta → focus on memory!
85-95%: stable range

Theta range observed: 0.500 to 0.900
CASR fixed: 0.800
```

---

## Project Structure

```
TASCAR/
├── config.py               ← All settings (CASR + TASCAR)
├── simulator.py            ← Azure dataset loader
├── scache.py               ← W-TinyLFU S-Cache (K=3)
├── environment.py          ← CASR RL environment
├── baselines.py            ← 5 baseline algorithms
├── ppo_agent.py            ← PPO agent (for CASR comparison)
├── transformer_encoder.py  ← Transformer + History buffer
├── sac_agent.py            ← SAC agent with 2 critics
├── train_tascar.py         ← TASCAR training script
├── evaluate_tascar.py      ← CASR vs TASCAR evaluation
├── evaluate.py             ← CASR full evaluation
├── results_tascar/         ← TASCAR comparison results
└── trained_model_tascar/   ← Trained TASCAR model
```

---

## Algorithm Details

### Transformer Encoder
```
Input:  Last 10 S-Cache states
        Shape: (10, 21)

Layers:
1. Linear projection: 21 → 64
2. Positional encoding
3. Transformer encoder (2 layers, 4 heads)
4. Cross-queue attention
5. Layer normalization
6. Output projection: 64 → 64

Output: Enriched state (64 numbers)
        Fed to SAC agent!
```

### SAC Agent
```
Actor:  64 → 128 → 128 → 27
        Outputs probabilities for 27 actions

Critic1: 64 → 128 → 128 → 27  ←┐
Critic2: 64 → 128 → 128 → 27  ←┘ Take minimum!

Entropy temperature (alpha):
Auto-tuned for optimal exploration!

Replay buffer: 100,000 experiences
Batch size: 64
Updates per step: 10
```

### Dynamic Reward
```
R = -(θ × R1_norm + (1-θ) × R2_norm)

Where:
R1 = cold starts this step
R2 = change in wasted memory time
θ  = dynamic theta (adapts automatically!)

CASR:   θ = 0.8 fixed always
TASCAR: θ adapts based on cold start rate!
```

---

## Training Details

```
Episodes:        500
Steps/episode:   100 (TASCAR_DELTA=1000)
Total steps:     50,000 (vs CASR 2,000)
Warmup episodes: 20 (fill buffer first!)
Best reward:     -0.1381
Final theta:     0.720

CASR comparison:
Episodes:      200
Steps/episode: 10 (DELTA=10000)
Total steps:   2,000
Best reward:   -0.0447
```

---

## Dataset

**Microsoft Azure Functions 2019**
- 1,332,032 function calls per day
- Queue 0 (0-1s):   124,663  (9.4%)
- Queue 1 (1-60s):  1,135,757 (85.3%)
- Queue 2 (60+s):   66,988   (5.0%)
- Download: https://github.com/Azure/AzurePublicDataset

---

## Installation

### Requirements
- Python 3.11
- Windows / Linux / Mac

### Setup Steps

**Step 1: Clone repository**
```
git clone https://github.com/Krishn4nmol/TASCAR.git
cd TASCAR
```

**Step 2: Activate environment**
```
cd ..\CASR_Project
casr_env\Scripts\activate
cd ..\TASCAR
```

**Step 3: Download Azure dataset**

Copy data folder from CASR_Project:
```
xcopy ..\CASR_Project\data data\ /E /I
```

**Step 4: Copy CASR trained model**
```
xcopy ..\CASR_Project\trained_model trained_model\ /E /I
```

---

## How to Run

### Train TASCAR
```
python train_tascar.py
```
Training takes approximately 15 minutes for 500 episodes.

### Compare CASR vs TASCAR
```
python evaluate_tascar.py
```
Evaluation takes approximately 1 hour with cooling breaks.

---

## Training Results

### TASCAR Convergence
```
Best reward:     -0.1381
Final theta:     0.720
Steps/episode:   100
Total steps:     50,000
Training time:   ~15 minutes
WMT throughout:  0.000s always
```

### CASR Convergence (for comparison)
```
Best reward:     -0.0447
Episodes:        200
Steps/episode:   10
Total steps:     2,000
Training time:   ~5 minutes
```

---

## Full Comparison Results

### Common Workload

| Algorithm | Cold% | WMT (s) | Overhead (s) |
|-----------|-------|---------|--------------|
| TASCAR | **71.493%** | **0.000** | **7.319** |
| CASR | 89.178% | 0.000 | 8.997 |
| Improvement | **-17.685 pp** | Same | **-1.678s** |

### Significant Workload

| Algorithm | Cold% | WMT (s) | Overhead (s) |
|-----------|-------|---------|--------------|
| TASCAR | **75.051%** | **0.000** | **8.621** |
| CASR | 91.449% | 0.000 | 10.248 |
| Improvement | **-16.398 pp** | Same | **-1.627s** |

### Random Workload

| Algorithm | Cold% | WMT (s) | Overhead (s) |
|-----------|-------|---------|--------------|
| TASCAR | **69.026%** | **0.000** | **8.255** |
| CASR | 85.138% | 0.000 | 9.626 |
| Improvement | **-16.112 pp** | Same | **-1.371s** |

---

## Implementation Environment

```
OS:        Windows 11
Processor: AMD Ryzen 7 8840HS
RAM:       32GB
Python:    3.11.9
PyTorch:   2.11.0
NumPy:     2.4.4
Gymnasium: 1.3.0
```

---

## Related Work

This project extends:

> Y. Chen, B. Liu, W. Lin, Y. Guo, and Z. Peng,
> "CASR: Optimizing cold start and resources utilization
> in serverless computing," Future Generation Computer
> Systems, vol. 170, p. 107851, 2025.

CASR implementation available at:
https://github.com/Krishn4nmol/CASR_Project

---

## Author

**Anmol Krishna**
Student Researcher
KIIT University, Bhubaneswar, India
GitHub: [Krishn4nmol](https://github.com/Krishn4nmol)
Email: anmolkrishna80@gmail.com

---

## Citation

```bibtex
@article{CHEN2025107851,
  title   = {CASR: Optimizing cold start and resources
             utilization in serverless computing},
  journal = {Future Generation Computer Systems},
  volume  = {170},
  pages   = {107851},
  year    = {2025},
  doi     = {10.1016/j.future.2025.107851},
  author  = {Yu Chen and Bo Liu and Weiwei Lin
             and Yulin Guo and Zhiping Peng}
}
```

---

## License

MIT License - Free to use and modify for research purposes.
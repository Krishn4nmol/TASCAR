# regenerate_figures.py
# Regenerates Fig 2 (mean±std bar chart)
# and Fig 3 (per-seed line chart)
# for all 7 seeds
# Uses hardcoded evaluation CSR values
# No model files needed!

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('results_figures', exist_ok=True)

# ─────────────────────────────────────────
# PER-SEED CSR VALUES
# From multiseed evaluation pipeline
# ─────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 1000, 2024, 2025]

# CASR per-seed CSR (%)
CASR = {
    'Common':      [90.737, 89.503, 90.525, 90.005, 91.146, 87.578, 87.543],
    'Significant': [91.892, 93.692, 93.625, 91.826, 94.302, 92.795, 95.735],
    'Random':      [85.675, 90.081, 77.012, 75.399, 78.905, 87.213, 86.113],
}

# TASCAR per-seed CSR (%)
TASCAR = {
    'Common':      [71.883, 71.848, 71.851, 71.381, 71.384, 71.381, 71.381],
    'Significant': [74.558, 74.558, 74.548, 75.048, 75.038, 75.048, 75.048],
    'Random':      [70.852, 70.838, 70.841, 71.019, 71.022, 71.019, 71.019],
}

WORKLOADS = ['Common', 'Significant', 'Random']

# ─────────────────────────────────────────
# FIG 2: Mean ± Std Bar Chart (7 seeds)
# Replaces old 3-seed version
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    'TASCAR vs CASR: Multi-Seed Validation\n'
    'Seeds: (42, 123, 456, 789, 1000, 2024, 2025) '
    '| Cold Start Rate (%) | Error bars = std',
    fontsize=11, fontweight='bold')

colors = {'CASR': '#2196F3', 'TASCAR': '#FF5722'}

for ax_idx, wl in enumerate(WORKLOADS):
    ax = axes[ax_idx]

    casr_vals   = np.array(CASR[wl])
    tascar_vals = np.array(TASCAR[wl])

    means  = [np.mean(casr_vals), np.mean(tascar_vals)]
    stds   = [np.std(casr_vals),  np.std(tascar_vals)]
    labels = ['CASR', 'TASCAR']
    cols   = [colors['CASR'], colors['TASCAR']]

    x    = np.arange(len(labels))
    bars = ax.bar(x, means, color=cols,
                  alpha=0.85, edgecolor='black',
                  linewidth=0.5)
    ax.errorbar(x, means, yerr=stds,
                fmt='none', color='black',
                capsize=8, linewidth=2, capthick=2)

    # Label bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.5,
                f'{mean:.1f}%\n±{std:.1f}',
                ha='center', fontsize=9,
                fontweight='bold')

    ax.set_title(f'{wl} Workload',
                 fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Cold Start Rate (%)')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path2 = 'results_figures/multiseed_comparison.png'
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved Fig 2: {path2}")

# ─────────────────────────────────────────
# FIG 3: Per-Seed TASCAR CSR (7 seeds)
# With mean dashed line
# ─────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle(
    'Per-Seed Cold Start Rate\n'
    'TASCAR across Seeds '
    '(42, 123, 456, 789, 1000, 2024, 2025)',
    fontsize=11, fontweight='bold')

# Use CASR TASCAR bar colors per seed
seed_colors = [
    '#FF5722', '#9C27B0', '#009688',
    '#F44336', '#3F51B5', '#FF9800', '#4CAF50']

seed_labels = [str(s) for s in SEEDS]

for ax_idx, wl in enumerate(WORKLOADS):
    ax = axes2[ax_idx]

    tascar_vals = [TASCAR[wl][i] for i in range(len(SEEDS))]
    casr_mean   = np.mean(CASR[wl])
    tascar_mean = np.mean(tascar_vals)

    x    = np.arange(len(SEEDS))
    bars = ax.bar(x, tascar_vals,
                  color=seed_colors,
                  alpha=0.85,
                  edgecolor='black',
                  linewidth=0.5)

    # Label each bar
    for bar, val in zip(bars, tascar_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val:.2f}%',
                ha='center', fontsize=7,
                fontweight='bold', rotation=45)

    # TASCAR mean line
    ax.axhline(y=tascar_mean, color='#FF5722',
               linestyle='--', linewidth=2,
               label=f'TASCAR Mean: {tascar_mean:.2f}%')

    # CASR mean line for reference
    ax.axhline(y=casr_mean, color='#2196F3',
               linestyle=':', linewidth=1.5,
               label=f'CASR Mean: {casr_mean:.2f}%')

    ax.set_title(f'{wl} Workload',
                 fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels, rotation=45, fontsize=8)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Cold Start Rate (%)')
    ax.set_ylim(60, 100)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path3 = 'results_figures/per_seed_comparison.png'
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved Fig 3: {path3}")

# ─────────────────────────────────────────
# PRINT STATS FOR VERIFICATION
# ─────────────────────────────────────────
print("\n=== Verification ===")
for wl in WORKLOADS:
    c = np.array(CASR[wl])
    t = np.array(TASCAR[wl])
    d = c - t
    print(f"{wl}:")
    print(f"  CASR:   {np.mean(c):.2f}±{np.std(c):.2f}%")
    print(f"  TASCAR: {np.mean(t):.2f}±{np.std(t):.2f}%")
    print(f"  Diff:   {np.mean(d):.2f}±{np.std(d):.2f}pp")

print("\nDone! Copy both files to your TASCAR folder.")
print("Fig 2: results_figures/multiseed_comparison.png")
print("Fig 3: results_figures/per_seed_comparison.png")
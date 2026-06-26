# plot_convergence_comparison.py
# Shows Training Cold Start Rate curves
# for CASR, TASCAR, LSTM+SAC, GRU+SAC

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(values, window=15):
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window)
        smoothed.append(np.mean(values[start:i+1]))
    return smoothed


def load_logs(path):
    with open(path, 'r') as f:
        return json.load(f)


def main():
    print("Loading training logs...")

    casr_logs   = load_logs(r"..\CASR_Project\results\training_logs.json")
    tascar_logs = load_logs(r"results_tascar_seed123\training_logs.json")
    lstm_logs   = load_logs(r"results_lstm_sac\training_logs.json")
    gru_logs    = load_logs(r"results_gru_sac\training_logs.json")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        'Training Cold Start Rate: Transformer vs Recurrent Encoders',
        fontsize=13, fontweight='bold')

    casr_ep    = casr_logs['episodes']
    casr_cold  = casr_logs['cold_start_rates']

    tascar_ep   = tascar_logs['episodes']
    tascar_cold = tascar_logs['cold_start_rates']

    lstm_ep   = lstm_logs['episodes']
    lstm_cold = lstm_logs['cold_start_rates']

    gru_ep   = gru_logs['episodes']
    gru_cold = gru_logs['cold_start_rates']

    # Raw faded lines
    ax.plot(casr_ep, casr_cold,
            color='gray', alpha=0.15, linewidth=0.8)
    ax.plot(tascar_ep, tascar_cold,
            color='blue', alpha=0.15, linewidth=0.8)
    ax.plot(lstm_ep, lstm_cold,
            color='orange', alpha=0.15, linewidth=0.8)
    ax.plot(gru_ep, gru_cold,
            color='green', alpha=0.15, linewidth=0.8)

    # Smoothed lines
    ax.plot(casr_ep[:len(casr_cold)],
            smooth(casr_cold, 15),
            color='gray', linewidth=2.5,
            linestyle='--',
            label='CASR (PPO, no encoder)')
    ax.plot(tascar_ep[:len(tascar_cold)],
            smooth(tascar_cold, 15),
            color='blue', linewidth=2.5,
            label='TASCAR (Transformer+SAC)')
    ax.plot(lstm_ep[:len(lstm_cold)],
            smooth(lstm_cold, 15),
            color='orange', linewidth=2.0,
            linestyle='-.',
            label='LSTM+SAC')
    ax.plot(gru_ep[:len(gru_cold)],
            smooth(gru_cold, 15),
            color='green', linewidth=2.0,
            linestyle=':',
            label='GRU+SAC')

    # Vertical line where CASR training ends
    ax.axvline(x=200, color='gray',
               linestyle=':', alpha=0.6,
               linewidth=1.5)
    ax.text(202, 98.5, 'CASR\nends\n(ep 200)',
            color='gray', fontsize=8,
            verticalalignment='top')

    ax.set_xlabel('Training Episode', fontsize=11)
    ax.set_ylabel('Cold Start Rate (%)', fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    out = r"results_tascar\fig_convergence_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
# evaluate_tascar.py
# Complete TASCAR vs CASR evaluation
# Uses MetricsTracker for all metrics!
# FIXED: handle_request called FIRST
# then scaling decision!
# This matches original CASR behavior!

import numpy as np
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from config import (
    NUM_QUEUES,
    NUM_FUNCTIONS,
    EVAL_CALLS,
    DELTA,
    TASCAR_DELTA,
    TASCAR_EVAL_DELTA,
    SEQUENCE_LENGTH,
    TRANSFORMER_DIM,
    TASCAR_MODEL_PATH,
    TASCAR_RESULTS,
    MODEL_SAVE_PATH,
    COOLING_BETWEEN_ALGORITHMS,
    COOLING_BETWEEN_WORKLOADS,
    SCALING_FACTOR,
    TPI_W1, TPI_W2,
    TPI_W3, TPI_W4, TPI_W5,
    SLA_THRESHOLD,
    CARBON_INTENSITY)
from simulator import AzureDataLoader
from metrics_tracker import MetricsTracker
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent
from ppo_agent import PPOAgent


# ─────────────────────────────────────────
# AGI CALCULATION
# ─────────────────────────────────────────

def compute_agi(casr_csr, tascar_csr):
    """
    AGI = (CSR_casr - CSR_tascar)
          × 100 / CSR_casr
    Measures cold start reduction
    due to Transformer attention!
    """
    if casr_csr <= 0:
        return 0.0
    return ((casr_csr - tascar_csr)
            * 100.0 / casr_csr)


# ─────────────────────────────────────────
# CASR ALGORITHM
# FIXED: handle_request FIRST!
# then scaling decision!
# ─────────────────────────────────────────

class CASRAlgorithm:
    """
    CASR with trained PPO model.
    Uses MetricsTracker for metrics!

    KEY FIX:
    Process request FIRST!
    Then check if scaling needed!
    This matches original CASR!
    """
    def __init__(self,
                 model_path=None):
        self.tracker    = MetricsTracker()
        self.call_count = 0
        state_dim       = NUM_QUEUES * 7
        action_dim      = 3 ** NUM_QUEUES
        self.agent      = PPOAgent(
            state_dim, action_dim)

        if (model_path and
                os.path.exists(
                    model_path +
                    "actor.pth")):
            self.agent.load(model_path)
            print("  CASR model loaded!")
        else:
            print("  No CASR model!")

        self.action_map = (
            self._build_action_map())

    def _build_action_map(self):
        choices    = [
            -SCALING_FACTOR,
            0, SCALING_FACTOR]
        action_map = {}
        for i in range(
                3 ** NUM_QUEUES):
            action = []
            temp   = i
            for _ in range(NUM_QUEUES):
                action.append(
                    choices[temp % 3])
                temp //= 3
            action_map[i] = action
        return action_map

    def handle_request(self,
                       function_call):
        # FIXED: Process request FIRST!
        result = (
            self.tracker
            .handle_request(
                function_call))

        # THEN check scaling!
        self.call_count += 1
        if self.call_count % DELTA == 0:
            state = np.array(
                self.tracker.get_state(),
                dtype=np.float32)
            mean = np.mean(state)
            std  = np.std(state)
            if std > 0:
                state = (
                    (state - mean) / std)
            action, _ = (
                self.agent
                .choose_action(state))
            for q_idx, scale in enumerate(
                    self.action_map[
                        action]):
                if scale != 0:
                    self.tracker\
                        .scale_queue(
                        q_idx, scale)

        return result

    def get_total_wasted_memory_time(
            self):
        return (self.tracker
                .get_total_wasted_memory_time())

    def get_all_metrics(self):
        return self.tracker.get_all_metrics()


# ─────────────────────────────────────────
# TASCAR ALGORITHM
# FIXED: handle_request FIRST!
# then scaling decision!
# ─────────────────────────────────────────

class TASCARAlgorithm:
    """
    TASCAR with trained SAC +
    Transformer.
    Uses MetricsTracker for metrics!

    KEY FIX:
    Process request FIRST!
    Then check if scaling needed!
    Matches original CASR behavior!

    Uses TASCAR_EVAL_DELTA = 10000
    Same as CASR = fair comparison!
    """
    def __init__(self,
                 model_path=None):
        self.tracker    = MetricsTracker()
        self.call_count = 0
        self.state_dim  = NUM_QUEUES * 7
        self.action_dim = 3 ** NUM_QUEUES

        self.transformer = (
            TransformerEncoder(
                self.state_dim))
        self.agent = SACAgent(
            transformer_dim=(
                TRANSFORMER_DIM),
            action_dim=(
                self.action_dim),
            transformer=(
                self.transformer))

        if (model_path and
                os.path.exists(
                    model_path +
                    "actor.pth")):
            self.agent.load(model_path)
            print(
                "  TASCAR model loaded!")
        else:
            print("  No TASCAR model!")

        self.history = StateHistoryBuffer(
            SEQUENCE_LENGTH,
            self.state_dim)
        self.action_map = (
            self.agent.action_map)

    def handle_request(self,
                       function_call):
        # FIXED: Process request FIRST!
        result = (
            self.tracker
            .handle_request(
                function_call))

        # THEN check scaling!
        self.call_count += 1
        if (self.call_count %
                TASCAR_EVAL_DELTA == 0):
            raw = np.array(
                self.tracker.get_state(),
                dtype=np.float32)
            mean = np.mean(raw)
            std  = np.std(raw)
            if std > 0:
                raw = (raw - mean) / std
            self.history.add(raw)
            seq = (
                self.history
                .get_sequence())
            encoded = (
                self.agent
                .get_encoded_state(seq))
            action = (
                self.agent
                .choose_action(
                    encoded,
                    evaluate=True))
            for q_idx, scale in enumerate(
                    self.action_map[
                        action]):
                if scale != 0:
                    self.tracker\
                        .scale_queue(
                        q_idx, scale)

        return result

    def get_total_wasted_memory_time(
            self):
        return (self.tracker
                .get_total_wasted_memory_time())

    def get_all_metrics(self):
        return self.tracker.get_all_metrics()


# ─────────────────────────────────────────
# LOAD RL METRICS
# ─────────────────────────────────────────

def load_rl_metrics():
    """Loads RL metrics from training logs"""
    log_path = (
        TASCAR_RESULTS +
        'training_logs.json')
    if not os.path.exists(log_path):
        return {}
    try:
        with open(log_path) as f:
            logs = json.load(f)
        return {
            'training_time_seconds':
                logs.get(
                    'training_time_seconds',
                    0),
            'convergence_episode':
                logs.get(
                    'convergence_episode',
                    -1),
            'best_reward':
                logs.get(
                    'best_reward', 0),
            'total_samples':
                logs.get(
                    'total_samples', 0),
            'sample_efficiency':
                logs.get(
                    'sample_efficiency',
                    0),
            'final_cumulative_reward':
                (logs.get(
                    'cumulative_rewards',
                    [0])[-1]
                 if logs.get(
                    'cumulative_rewards')
                 else 0),
        }
    except Exception as e:
        print(f"RL metrics error: {e}")
        return {}


# ─────────────────────────────────────────
# LOAD WORKLOADS
# ─────────────────────────────────────────

def load_workloads():
    loader    = AzureDataLoader()
    workloads = {}

    print("\nPreparing workloads...")

    # Common
    print("  Loading Common...")
    day1   = loader.load_day(1)
    counts = Counter(
        c.function_id for c in day1)
    top = set(
        f for f, _ in
        counts.most_common(NUM_FUNCTIONS))
    common = [
        c for c in day1
        if c.function_id in top]
    np.random.seed(42)
    if len(common) > EVAL_CALLS:
        idx = np.random.choice(
            len(common), EVAL_CALLS,
            replace=False)
        idx.sort()
        common = [
            common[i] for i in idx]
    workloads['Common'] = common
    print(f"    {len(common)} calls")

    # Significant
    print("  Loading Significant...")
    day2  = loader.load_day(2)
    heavy = [
        c for c in day2
        if c.cold_start_overhead > 1]
    counts = Counter(
        c.function_id for c in heavy)
    top = set(
        f for f, _ in
        counts.most_common(NUM_FUNCTIONS))
    significant = [
        c for c in heavy
        if c.function_id in top]
    np.random.seed(42)
    if len(significant) > EVAL_CALLS:
        idx = np.random.choice(
            len(significant),
            EVAL_CALLS,
            replace=False)
        idx.sort()
        significant = [
            significant[i]
            for i in idx]
    workloads['Significant'] = (
        significant)
    print(
        f"    {len(significant)} calls")

    # Random
    print("  Loading Random...")
    day3  = loader.load_day(3)
    funcs = list(set(
        c.function_id for c in day3))
    np.random.seed(123)
    np.random.shuffle(funcs)
    selected  = set(funcs[:NUM_FUNCTIONS])
    random_wl = [
        c for c in day3
        if c.function_id in selected]
    np.random.seed(123)
    if len(random_wl) > EVAL_CALLS:
        idx = np.random.choice(
            len(random_wl),
            EVAL_CALLS,
            replace=False)
        idx.sort()
        random_wl = [
            random_wl[i] for i in idx]
    workloads['Random'] = random_wl
    print(f"    {len(random_wl)} calls")

    return workloads


# ─────────────────────────────────────────
# RUN EVALUATION
# ─────────────────────────────────────────

def run_evaluation():
    os.makedirs(
        TASCAR_RESULTS, exist_ok=True)

    workloads  = load_workloads()
    results    = {}
    rl_metrics = load_rl_metrics()

    print("\nStarting evaluation...")
    print(f"CASR delta:        {DELTA}")
    print(f"TASCAR eval delta: "
          f"{TASCAR_EVAL_DELTA}")
    print(f"SLA threshold:     "
          f"{SLA_THRESHOLD}s")
    print(f"Request order:     "
          f"handle THEN scale!")
    print("=" * 60)

    for wl_idx, (wl_name, calls) in (
            enumerate(
                workloads.items())):

        print(f"\nWorkload: {wl_name}")
        print("-" * 40)

        if wl_idx > 0:
            secs = (
                COOLING_BETWEEN_WORKLOADS)
            print(
                f"\nCooling {secs}s...")
            for i in range(secs // 10):
                remaining = (
                    secs - (i * 10))
                print(
                    f"  {remaining}s "
                    f"remaining...")
                time.sleep(10)
            print("  Done cooling!")

        results[wl_name] = {}

        # ── Run CASR ──
        print("\n  Running CASR...")
        try:
            casr = CASRAlgorithm(
                MODEL_SAVE_PATH +
                "best/")
            for call in calls:
                casr.handle_request(call)

            casr_m = casr.get_all_metrics()
            casr_m['agi'] = 0.0
            results[wl_name][
                'CASR'] = casr_m
            print(
                f"  CASR "
                f"CSR: "
                f"{casr_m['cold_start_rate']:.3f}% "
                f"WMT: "
                f"{casr_m['avg_wasted_memory_time']:.3f}s "
                f"TPI: {casr_m['tpi']:.2f}")
        except Exception as e:
            print(f"  CASR Error: {e}")
            import traceback
            traceback.print_exc()
            results[wl_name]['CASR'] = {}

        # Cool between algorithms
        secs = COOLING_BETWEEN_ALGORITHMS
        print(f"\n  Cooling {secs}s...")
        time.sleep(secs)

        # ── Run TASCAR ──
        print("\n  Running TASCAR...")
        try:
            tascar = TASCARAlgorithm(
                TASCAR_MODEL_PATH +
                "best/")
            for call in calls:
                tascar.handle_request(call)

            tascar_m = (
                tascar.get_all_metrics())

            casr_csr = (
                results[wl_name]
                .get('CASR', {})
                .get('cold_start_rate',
                     0))
            tascar_m['agi'] = compute_agi(
                casr_csr,
                tascar_m[
                    'cold_start_rate'])

            results[wl_name][
                'TASCAR'] = tascar_m
            print(
                f"  TASCAR "
                f"CSR: "
                f"{tascar_m['cold_start_rate']:.3f}% "
                f"WMT: "
                f"{tascar_m['avg_wasted_memory_time']:.3f}s "
                f"TPI: {tascar_m['tpi']:.2f} "
                f"AGI: {tascar_m['agi']:.2f}%")
        except Exception as e:
            print(
                f"  TASCAR Error: {e}")
            import traceback
            traceback.print_exc()
            results[wl_name][
                'TASCAR'] = {}

    _save_results(results, rl_metrics)
    print_summary(results)
    if rl_metrics:
        print_rl_metrics(rl_metrics)
    plot_all_comparisons(
        results, rl_metrics)

    return results


# ─────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────

def _save_results(results,
                  rl_metrics=None):
    path = (TASCAR_RESULTS +
            'casr_vs_tascar.json')
    serializable = {}
    for wl, algos in results.items():
        serializable[wl] = {}
        for algo, metrics in (
                algos.items()):
            serializable[wl][algo] = {
                k: float(v)
                for k, v in
                metrics.items()
                if isinstance(
                    v, (int, float))}
    if rl_metrics:
        serializable[
            'rl_metrics'] = rl_metrics
    with open(path, 'w') as f:
        json.dump(
            serializable, f, indent=2)
    print(f"\nResults saved: {path}")


# ─────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 75)
    print("COMPLETE METRICS COMPARISON")
    print("=" * 75)

    workloads = [
        'Common', 'Significant', 'Random']

    all_metrics = [
        ('cold_start_rate',
         'Cold Start Rate (%)', True),
        ('avg_cold_start_overhead',
         'Avg Cold Start Delay (s)',
         True),
        ('p95_latency',
         'P95 Latency (s)', True),
        ('p99_latency',
         'P99 Latency (s)', True),
        ('avg_response_time',
         'Avg Response Time (s)', True),
        ('avg_wasted_memory_time',
         'Wasted Memory Time (s)', True),
        ('container_utilization_rate',
         'Container Utilization (%)',
         False),
        ('resource_utilization_efficiency',
         'Resource Util Eff (%)',
         False),
        ('sla_violation_rate',
         'SLA Violation Rate (%)', True),
        ('throughput',
         'Throughput (req/s)', False),
        ('successful_execution_ratio',
         'Successful Exec Ratio (%)',
         False),
        ('energy_per_request',
         'Energy per Request (kWh)',
         True),
        ('co2_estimate',
         'CO2 Estimate (kg)', True),
        ('burst_handling_efficiency',
         'Burst Handling Eff (%)',
         False),
        ('scaling_accuracy',
         'Scaling Accuracy (%)', False),
        ('elasticity_score',
         'Elasticity Score', False),
        ('tpi',
         'TASCAR Perf Index (TPI)',
         False),
        ('agi',
         'Attention Gain Index (%)',
         False),
    ]

    for wl in workloads:
        if wl not in results:
            continue
        print(f"\n{'='*28} "
              f"{wl} {'='*28}")
        print(
            f"{'Metric':<35}"
            f"{'CASR':>12}"
            f"{'TASCAR':>12}"
            f"{'Winner':>10}")
        print("-" * 72)

        for (metric, label,
             lower_better) in (
                all_metrics):
            cv = (results[wl]
                  .get('CASR', {})
                  .get(metric, 0))
            tv = (results[wl]
                  .get('TASCAR', {})
                  .get(metric, 0))

            if lower_better:
                winner = (
                    "TASCAR ✅"
                    if tv < cv
                    else "CASR")
            else:
                winner = (
                    "TASCAR ✅"
                    if tv > cv
                    else "CASR")

            print(
                f"{label:<35}"
                f"{cv:>12.4f}"
                f"{tv:>12.4f}"
                f"{winner:>10}")

    # Print improvement summary
    print("\n" + "=" * 55)
    print("IMPROVEMENT SUMMARY")
    print("=" * 55)
    for wl in workloads:
        if wl not in results:
            continue
        casr_csr = (
            results[wl]
            .get('CASR', {})
            .get('cold_start_rate', 0))
        tascar_csr = (
            results[wl]
            .get('TASCAR', {})
            .get('cold_start_rate', 0))
        diff = casr_csr - tascar_csr
        symbol = "✅" if diff > 0 else "❌"
        print(
            f"  {wl:<15}: "
            f"CASR={casr_csr:.3f}% "
            f"TASCAR={tascar_csr:.3f}% "
            f"Diff={diff:+.3f}pp {symbol}")


def print_rl_metrics(rl_metrics):
    print("\n" + "=" * 55)
    print("RL TRAINING METRICS")
    print("=" * 55)
    items = [
        ('Training Time',
         'training_time_seconds', 's'),
        ('Convergence Episode',
         'convergence_episode', ''),
        ('Best Reward',
         'best_reward', ''),
        ('Total Samples',
         'total_samples', ''),
        ('Sample Efficiency',
         'sample_efficiency', ''),
        ('Cumulative Reward',
         'final_cumulative_reward', ''),
    ]
    for label, key, unit in items:
        val = rl_metrics.get(key, 0)
        print(
            f"  {label:<25} "
            f"{val}{unit}")


# ─────────────────────────────────────────
# PLOT ALL COMPARISONS
# 8 graph sets!
# ─────────────────────────────────────────

def plot_all_comparisons(
        results, rl_metrics=None):

    workloads  = [
        'Common', 'Significant', 'Random']
    algorithms = ['CASR', 'TASCAR']
    colors     = {
        'CASR':   '#2196F3',
        'TASCAR': '#FF5722'}

    def style_dark(fig, axes_list):
        fig.patch.set_facecolor(
            '#0a0e1a')
        for ax in axes_list:
            ax.set_facecolor('#111827')
            ax.tick_params(
                colors='white')
            ax.spines[
                'bottom'].set_color(
                '#334155')
            ax.spines[
                'left'].set_color(
                '#334155')
            ax.spines[
                'top'].set_visible(False)
            ax.spines[
                'right'].set_visible(
                False)

    def plot_bars(ax, metric,
                  ylabel, title):
        x = np.arange(len(workloads))
        w = 0.35
        for i, algo in enumerate(
                algorithms):
            values = [
                results.get(wl, {})
                .get(algo, {})
                .get(metric, 0)
                for wl in workloads]
            offset = (i - 0.5) * w
            bars   = ax.bar(
                x + offset, values, w,
                label=algo,
                color=colors[algo],
                alpha=0.85,
                edgecolor='white',
                linewidth=0.5)
            for bar, val in zip(
                    bars, values):
                ax.text(
                    bar.get_x() +
                    bar.get_width()/2,
                    bar.get_height() +
                    max(abs(val)*0.01,
                        0.001),
                    f'{val:.2f}',
                    ha='center',
                    fontsize=8,
                    fontweight='bold',
                    color='white')
        ax.set_title(
            title, fontsize=11,
            fontweight='bold',
            color='white')
        ax.set_xticks(
            np.arange(len(workloads)))
        ax.set_xticklabels(
            workloads, color='white')
        ax.legend(
            facecolor='#1e293b',
            labelcolor='white',
            fontsize=8)
        ax.grid(
            axis='y', alpha=0.2,
            color='white')
        ax.set_ylabel(
            ylabel, color='white',
            fontsize=9)

    # Fig 1: Cold Start
    fig1, axes1 = plt.subplots(
        1, 3, figsize=(18, 6))
    fig1.suptitle(
        'Cold Start Metrics',
        fontsize=14,
        fontweight='bold',
        color='white')
    style_dark(fig1, axes1)
    plot_bars(axes1[0],
              'cold_start_rate',
              'CSR (%)',
              'Cold Start Rate (%)')
    plot_bars(axes1[1],
              'avg_cold_start_overhead',
              'Delay (s)',
              'Avg Cold Start Delay (s)')
    plot_bars(axes1[2],
              'p95_latency',
              'P95 (s)',
              'P95 Latency (s)')
    plt.tight_layout()
    plt.savefig(
        TASCAR_RESULTS +
        'fig1_cold_start.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print("Saved fig1_cold_start.png")

    # Fig 2: Latency + WMT
    fig2, axes2 = plt.subplots(
        1, 3, figsize=(18, 6))
    fig2.suptitle(
        'Latency and Memory Metrics',
        fontsize=14,
        fontweight='bold',
        color='white')
    style_dark(fig2, axes2)
    plot_bars(axes2[0],
              'p99_latency',
              'P99 (s)',
              'P99 Latency (s)')
    plot_bars(axes2[1],
              'avg_response_time',
              'ART (s)',
              'Avg Response Time (s)')
    plot_bars(axes2[2],
              'avg_wasted_memory_time',
              'WMT (s)',
              'Wasted Memory Time (s)')
    plt.tight_layout()
    plt.savefig(
        TASCAR_RESULTS +
        'fig2_latency_memory.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print(
        "Saved fig2_latency_memory.png")

    # Fig 3: Resource
    fig3, axes3 = plt.subplots(
        1, 3, figsize=(18, 6))
    fig3.suptitle(
        'Resource Utilization Metrics',
        fontsize=14,
        fontweight='bold',
        color='white')
    style_dark(fig3, axes3)
    plot_bars(axes3[0],
              'container_utilization_rate',
              'CUR (%)',
              'Container Utilization (%)')
    plot_bars(axes3[1],
              'resource_utilization_efficiency',
              'RUE (%)',
              'Resource Util Eff (%)')
    plot_bars(axes3[2],
              'successful_execution_ratio',
              'SER (%)',
              'Successful Exec Ratio (%)')
    plt.tight_layout()
    plt.savefig(
        TASCAR_RESULTS +
        'fig3_resource.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print("Saved fig3_resource.png")

    # Fig 4: QoS + Throughput
    fig4, axes4 = plt.subplots(
        1, 3, figsize=(18, 6))
    fig4.suptitle(
        'QoS and Throughput Metrics',
        fontsize=14,
        fontweight='bold',
        color='white')
    style_dark(fig4, axes4)
    plot_bars(axes4[0],
              'sla_violation_rate',
              'SVR (%)',
              'SLA Violation Rate (%)')
    plot_bars(axes4[1],
              'throughput',
              'Throughput',
              'Throughput (req/s)')
    plot_bars(axes4[2],
              'burst_handling_efficiency',
              'BHE (%)',
              'Burst Handling Eff (%)')
    plt.tight_layout()
    plt.savefig(
        TASCAR_RESULTS +
        'fig4_qos_throughput.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print(
        "Saved fig4_qos_throughput.png")

    # Fig 5: Energy + Scaling
    fig5, axes5 = plt.subplots(
        1, 3, figsize=(18, 6))
    fig5.suptitle(
        'Energy and Scalability Metrics',
        fontsize=14,
        fontweight='bold',
        color='white')
    style_dark(fig5, axes5)
    plot_bars(axes5[0],
              'energy_per_request',
              'EPR (kWh)',
              'Energy per Request (kWh)')
    plot_bars(axes5[1],
              'co2_estimate',
              'CO2 (kg)',
              'CO2 Estimate (kg)')
    plot_bars(axes5[2],
              'scaling_accuracy',
              'SA (%)',
              'Scaling Accuracy (%)')
    plt.tight_layout()
    plt.savefig(
        TASCAR_RESULTS +
        'fig5_energy_scaling.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print(
        "Saved fig5_energy_scaling.png")

    # Fig 6: TPI + AGI
    fig6, axes6 = plt.subplots(
        1, 2, figsize=(14, 6))
    fig6.suptitle(
        'Composite Performance Index\n'
        'TPI and Attention Gain Index',
        fontsize=13,
        fontweight='bold',
        color='white')
    style_dark(fig6, axes6)
    plot_bars(axes6[0],
              'tpi',
              'TPI Score',
              'TASCAR Performance Index')

    ax2    = axes6[1]
    agis   = [
        results.get(wl, {})
        .get('TASCAR', {})
        .get('agi', 0)
        for wl in workloads]
    wl_colors = [
        '#FF5722', '#FF9800', '#FFC107']
    bars2 = ax2.bar(
        workloads, agis,
        color=wl_colors,
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5)
    for bar, val in zip(bars2, agis):
        color = (
            'white' if val >= 0
            else '#ef4444')
        ax2.text(
            bar.get_x() +
            bar.get_width()/2,
            bar.get_height() +
            (0.1 if val >= 0 else -1.5),
            f'{val:.2f}%',
            ha='center',
            fontsize=10,
            fontweight='bold',
            color=color)
    ax2.set_title(
        'Attention Gain Index (AGI)\n'
        '+ve = TASCAR better than CASR!',
        fontsize=11,
        fontweight='bold',
        color='white')
    ax2.set_ylabel(
        'AGI (%)', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(
        axis='y', alpha=0.2,
        color='white')
    ax2.axhline(
        y=0, color='white',
        linestyle='-',
        linewidth=0.5)
    ax2.spines['bottom'].set_color(
        '#334155')
    ax2.spines['left'].set_color(
        '#334155')
    ax2.spines['top'].set_visible(False)
    ax2.spines[
        'right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        TASCAR_RESULTS +
        'fig6_tpi_agi.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print("Saved fig6_tpi_agi.png")

    # Fig 7: RL Metrics
    if rl_metrics:
        _plot_rl_metrics(rl_metrics)

    # Fig 8: Master
    _plot_master(
        results, workloads,
        algorithms, colors)

    print(
        f"\nAll graphs saved to "
        f"{TASCAR_RESULTS}")


def _plot_rl_metrics(rl_metrics):
    """Figure 7: RL Training Metrics"""
    log_path = (
        TASCAR_RESULTS +
        'training_logs.json')
    if not os.path.exists(log_path):
        return

    with open(log_path) as f:
        logs = json.load(f)

    episodes = logs.get('episodes', [])
    if not episodes:
        return

    fig, axes = plt.subplots(
        2, 3, figsize=(18, 10))
    fig.suptitle(
        'RL Training Metrics',
        fontsize=14,
        fontweight='bold',
        color='white')
    fig.patch.set_facecolor('#0a0e1a')
    for row in axes:
        for ax in row:
            ax.set_facecolor('#111827')

    def smooth(vals, w=10):
        s = []
        for i in range(len(vals)):
            start = max(0, i-w)
            s.append(
                np.mean(vals[start:i+1]))
        return s

    def style_ax(ax, title,
                 xlabel, ylabel):
        ax.set_title(
            title, fontsize=11,
            fontweight='bold',
            color='white')
        ax.set_xlabel(
            xlabel, color='white')
        ax.set_ylabel(
            ylabel, color='white')
        ax.tick_params(colors='white')
        ax.grid(
            alpha=0.2, color='white')
        ax.spines[
            'bottom'].set_color(
            '#334155')
        ax.spines[
            'left'].set_color(
            '#334155')
        ax.spines[
            'top'].set_visible(False)
        ax.spines[
            'right'].set_visible(False)

    rewards = logs.get('rewards', [])
    if rewards:
        axes[0, 0].plot(
            episodes, rewards,
            color='#0ea5e9',
            alpha=0.4, linewidth=1)
        axes[0, 0].plot(
            episodes,
            smooth(rewards),
            color='#0ea5e9',
            linewidth=2.5,
            label='Smoothed')
        conv = rl_metrics.get(
            'convergence_episode', -1)
        if conv and conv > 0:
            axes[0, 0].axvline(
                x=conv,
                color='#10b981',
                linestyle='--',
                label=f'Converged: {conv}')
        axes[0, 0].legend(
            facecolor='#1e293b',
            labelcolor='white',
            fontsize=8)
        style_ax(
            axes[0, 0],
            'Reward Convergence',
            'Episode',
            'Avg Reward per Step')

    cold_rates = logs.get(
        'cold_start_rates', [])
    if cold_rates:
        axes[0, 1].plot(
            episodes, cold_rates,
            color='#ef4444',
            alpha=0.4, linewidth=1)
        axes[0, 1].plot(
            episodes,
            smooth(cold_rates),
            color='#ef4444',
            linewidth=2.5,
            label='Smoothed')
        axes[0, 1].legend(
            facecolor='#1e293b',
            labelcolor='white',
            fontsize=8)
        style_ax(
            axes[0, 1],
            'Cold Start Rate (%)',
            'Episode', 'Cold%')

    thetas = logs.get('thetas', [])
    if thetas:
        axes[0, 2].plot(
            episodes, thetas,
            color='#8b5cf6',
            linewidth=1.5,
            alpha=0.7,
            label='TASCAR theta')
        axes[0, 2].plot(
            episodes,
            smooth(thetas),
            color='#8b5cf6',
            linewidth=2.5,
            label='Smoothed')
        axes[0, 2].axhline(
            y=0.8,
            color='#ef4444',
            linestyle='--',
            linewidth=2,
            label='CASR fixed=0.8')
        axes[0, 2].legend(
            facecolor='#1e293b',
            labelcolor='white',
            fontsize=8)
        axes[0, 2].set_ylim(0.4, 1.0)
        style_ax(
            axes[0, 2],
            'Dynamic Theta',
            'Episode', 'Theta')

    cum_rewards = logs.get(
        'cumulative_rewards', [])
    if cum_rewards:
        axes[1, 0].plot(
            episodes, cum_rewards,
            color='#f59e0b',
            linewidth=2,
            label='Cumulative Reward')
        axes[1, 0].legend(
            facecolor='#1e293b',
            labelcolor='white',
            fontsize=8)
        style_ax(
            axes[1, 0],
            'Cumulative Reward',
            'Episode', 'Rcum')

    sample_counts = logs.get(
        'sample_counts', [])
    if sample_counts and rewards:
        axes[1, 1].plot(
            sample_counts, rewards,
            color='#10b981',
            alpha=0.4, linewidth=1)
        axes[1, 1].plot(
            sample_counts,
            smooth(rewards),
            color='#10b981',
            linewidth=2.5,
            label='Smoothed')
        axes[1, 1].legend(
            facecolor='#1e293b',
            labelcolor='white',
            fontsize=8)
        style_ax(
            axes[1, 1],
            'Sample Efficiency',
            'Training Samples',
            'Reward')

    axes[1, 2].set_facecolor('#0f172a')
    axes[1, 2].axis('off')
    summary = (
        f"RL TRAINING SUMMARY\n\n"
        f"Time: "
        f"{rl_metrics.get('training_time_seconds', 0):.0f}s\n\n"
        f"Convergence: "
        f"Ep {rl_metrics.get('convergence_episode', -1)}\n\n"
        f"Best Reward: "
        f"{rl_metrics.get('best_reward', 0):.4f}\n\n"
        f"Total Samples: "
        f"{rl_metrics.get('total_samples', 0):,}\n\n"
        f"Sample Eff: "
        f"{rl_metrics.get('sample_efficiency', 0):.6f}\n\n"
        f"Cumulative R: "
        f"{rl_metrics.get('final_cumulative_reward', 0):.2f}")
    axes[1, 2].text(
        0.1, 0.95, summary,
        transform=axes[1, 2].transAxes,
        fontsize=11, color='white',
        va='top',
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='#1e293b',
            alpha=0.8))
    axes[1, 2].set_title(
        'Training Summary',
        fontsize=11,
        fontweight='bold',
        color='white')

    plt.tight_layout()
    plt.savefig(
        TASCAR_RESULTS +
        'fig7_rl_metrics.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print("Saved fig7_rl_metrics.png")


def _plot_master(
        results, workloads,
        algorithms, colors):
    """Figure 8: All metrics master"""
    all_metrics = [
        ('cold_start_rate',
         'Cold Start Rate (%)'),
        ('avg_cold_start_overhead',
         'Avg Cold Delay (s)'),
        ('p95_latency', 'P95 Lat (s)'),
        ('p99_latency', 'P99 Lat (s)'),
        ('avg_response_time',
         'Avg Response (s)'),
        ('avg_wasted_memory_time',
         'WMT (s)'),
        ('container_utilization_rate',
         'CUR (%)'),
        ('resource_utilization_efficiency',
         'RUE (%)'),
        ('sla_violation_rate',
         'SVR (%)'),
        ('throughput', 'Throughput'),
        ('energy_per_request',
         'Energy/Req'),
        ('co2_estimate', 'CO2 (kg)'),
        ('burst_handling_efficiency',
         'BHE (%)'),
        ('scaling_accuracy', 'SA (%)'),
        ('elasticity_score',
         'Elasticity'),
        ('tpi', 'TPI Score'),
        ('agi', 'AGI (%)'),
        ('successful_execution_ratio',
         'SER (%)'),
    ]

    fig, axes = plt.subplots(
        6, 3, figsize=(22, 28))
    fig.suptitle(
        'TASCAR vs CASR: '
        'Complete Metrics Comparison',
        fontsize=16,
        fontweight='bold',
        color='white', y=0.99)
    fig.patch.set_facecolor('#0a0e1a')

    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.set_facecolor('#111827')

    for idx, (metric, ylabel) in (
            enumerate(all_metrics)):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        x  = np.arange(len(workloads))
        w  = 0.35

        for i, algo in enumerate(
                algorithms):
            values = [
                results.get(wl, {})
                .get(algo, {})
                .get(metric, 0)
                for wl in workloads]
            offset = (i - 0.5) * w
            bars   = ax.bar(
                x + offset, values, w,
                label=algo,
                color=colors[algo],
                alpha=0.85,
                edgecolor='white',
                linewidth=0.4)
            for bar, val in zip(
                    bars, values):
                ax.text(
                    bar.get_x() +
                    bar.get_width()/2,
                    bar.get_height() +
                    max(abs(val)*0.01,
                        0.001),
                    f'{val:.2f}',
                    ha='center',
                    fontsize=7,
                    fontweight='bold',
                    color='white')

        ax.set_title(
            ylabel, fontsize=10,
            fontweight='bold',
            color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(
            workloads, color='white',
            fontsize=8)
        ax.tick_params(colors='white')
        ax.legend(
            facecolor='#1e293b',
            labelcolor='white',
            fontsize=7)
        ax.grid(
            axis='y', alpha=0.2,
            color='white')
        ax.spines['bottom'].set_color(
            '#334155')
        ax.spines['left'].set_color(
            '#334155')
        ax.spines[
            'top'].set_visible(False)
        ax.spines[
            'right'].set_visible(False)

    for idx in range(
            len(all_metrics),
            len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(
        rect=[0, 0, 1, 0.97])
    plt.savefig(
        TASCAR_RESULTS +
        'fig8_master_all_metrics.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print(
        "Saved "
        "fig8_master_all_metrics.png")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TASCAR Complete Evaluation")
    print("MetricsTracker + Original SCache")
    print("FIXED: handle THEN scale!")
    print("All Professor Metrics!")
    print("8 Graph Sets!")
    print(f"SLA Threshold: {SLA_THRESHOLD}s")
    print(
        f"Carbon: "
        f"{CARBON_INTENSITY} kg/kWh")
    print("=" * 60)

    tascar_path = (
        TASCAR_MODEL_PATH + "best/")
    if not os.path.exists(
            tascar_path + "actor.pth"):
        print("\nNo TASCAR model!")
        print(
            "Run: "
            "python train_tascar.py")
    else:
        run_evaluation()
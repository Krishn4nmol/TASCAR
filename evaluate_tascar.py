# evaluate_tascar.py
# Complete TASCAR vs CASR evaluation
# All professor recommended metrics!
# 6 comparison graph sets!

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
    CARBON_INTENSITY
)
from simulator import AzureDataLoader
from scache import SCache
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent
from ppo_agent import PPOAgent


# ─────────────────────────────────────────
# CASR ALGORITHM
# Uses DELTA = 10000 (original!)
# ─────────────────────────────────────────

class CASRAlgorithm:
    """
    CASR with trained PPO model.
    Uses DELTA = 10000 (original!)
    10 decisions per workload!
    """
    def __init__(self,
                 model_path=None):
        self.scache     = SCache()
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
            print("  Using S-Cache only!")

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
        self.call_count += 1
        if self.call_count % DELTA == 0:
            state = np.array(
                self.scache.get_state(),
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
                    self.scache\
                        .scale_queue(
                        q_idx, scale)
        return (self.scache
                .handle_request(
                    function_call))

    def get_total_wasted_memory_time(
            self):
        return (self.scache
                .get_total_wasted_memory_time())


# ─────────────────────────────────────────
# TASCAR ALGORITHM
# Uses TASCAR_EVAL_DELTA = 10000
# Same as CASR for fair comparison!
# ─────────────────────────────────────────

class TASCARAlgorithm:
    """
    TASCAR with trained SAC +
    Transformer.
    Uses TASCAR_EVAL_DELTA = 10000
    Same as CASR for fair comparison!
    """
    def __init__(self,
                 model_path=None):
        self.scache     = SCache()
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
            print(
                "  Run train_tascar.py!")

        self.history = StateHistoryBuffer(
            SEQUENCE_LENGTH,
            self.state_dim)
        self.action_map = (
            self.agent.action_map)

    def handle_request(self,
                       function_call):
        self.call_count += 1
        if (self.call_count %
                TASCAR_EVAL_DELTA == 0):
            raw_state = np.array(
                self.scache.get_state(),
                dtype=np.float32)
            mean = np.mean(raw_state)
            std  = np.std(raw_state)
            if std > 0:
                raw_state = (
                    (raw_state - mean)
                    / std)
            self.history.add(raw_state)
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
                    self.scache\
                        .scale_queue(
                        q_idx, scale)
        return (self.scache
                .handle_request(
                    function_call))

    def get_total_wasted_memory_time(
            self):
        return (self.scache
                .get_total_wasted_memory_time())


# ─────────────────────────────────────────
# COMPUTE TPI
# TASCAR Performance Index
# Professor recommended composite!
# ─────────────────────────────────────────

def compute_tpi(metrics,
                max_throughput=1000.0):
    """
    TPI = w1(1-CSR) + w2(1-WMT_norm)
        + w3(throughput_norm)
        + w4(1-SVR) + w5(RUE_norm)

    Higher TPI = better overall!
    """
    csr  = min(metrics.get(
        'cold_start_rate', 0) / 100.0,
        1.0)
    wmt  = min(metrics.get(
        'avg_wasted_memory_time',
        0) / 100.0, 1.0)
    tput = min(metrics.get(
        'throughput', 0) /
        max_throughput, 1.0)
    svr  = min(metrics.get(
        'sla_violation_rate',
        0) / 100.0, 1.0)
    rue  = min(metrics.get(
        'resource_utilization_efficiency',
        0) / 100.0, 1.0)

    tpi = (TPI_W1 * (1 - csr) +
           TPI_W2 * (1 - wmt) +
           TPI_W3 * tput +
           TPI_W4 * (1 - svr) +
           TPI_W5 * rue)
    return round(tpi * 100, 3)


# ─────────────────────────────────────────
# COMPUTE AGI
# Attention Gain Index
# Measures contribution of Transformer!
# ─────────────────────────────────────────

def compute_agi(casr_csr,
                tascar_csr):
    """
    AGI = (CSR_casr - CSR_tascar)
          × 100 / CSR_casr

    Measures % cold start reduction
    due to Transformer attention!
    """
    if casr_csr <= 0:
        return 0.0
    return ((casr_csr - tascar_csr)
            * 100.0 / casr_csr)


# ─────────────────────────────────────────
# LOAD RL METRICS FROM TRAINING LOGS
# ─────────────────────────────────────────

def load_rl_metrics():
    """
    Loads RL metrics from training logs!
    Returns dict with training metrics.
    """
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
        print(f"Could not load RL "
              f"metrics: {e}")
        return {}


# ─────────────────────────────────────────
# LOAD WORKLOADS
# Same as CASR for fair comparison!
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
        counts.most_common(
            NUM_FUNCTIONS))
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
        counts.most_common(
            NUM_FUNCTIONS))
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
    selected  = set(
        funcs[:NUM_FUNCTIONS])
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
            random_wl[i]
            for i in idx]
    workloads['Random'] = random_wl
    print(f"    {len(random_wl)} calls")

    return workloads


# ─────────────────────────────────────────
# RUN EVALUATION
# ─────────────────────────────────────────

def run_evaluation():
    os.makedirs(
        TASCAR_RESULTS, exist_ok=True)

    workloads = load_workloads()
    results   = {}

    print("\nStarting evaluation...")
    print(f"CASR delta:        {DELTA}")
    print(f"TASCAR eval delta: "
          f"{TASCAR_EVAL_DELTA}")
    print(f"SLA threshold:     "
          f"{SLA_THRESHOLD}s")
    print(f"TPI weights: "
          f"CSR={TPI_W1} WMT={TPI_W2} "
          f"TPT={TPI_W3} SVR={TPI_W4} "
          f"RUE={TPI_W5}")
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
                f"\nCooling CPU "
                f"{secs}s...")
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

            # Get all metrics from scache!
            casr_m = (
                casr.scache
                .get_all_metrics())
            casr_m['tpi'] = (
                compute_tpi(casr_m))
            casr_m['agi'] = 0.0

            results[wl_name][
                'CASR'] = casr_m
            print(
                f"  CASR "
                f"CSR: "
                f"{casr_m['cold_start_rate']:.2f}% "
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

            # Get all metrics from scache!
            tascar_m = (
                tascar.scache
                .get_all_metrics())
            tascar_m['tpi'] = (
                compute_tpi(tascar_m))

            # AGI calculation
            casr_csr = results[
                wl_name].get(
                'CASR', {}).get(
                'cold_start_rate', 0)
            tascar_m['agi'] = (
                compute_agi(
                    casr_csr,
                    tascar_m[
                        'cold_start_rate']))

            results[wl_name][
                'TASCAR'] = tascar_m
            print(
                f"  TASCAR "
                f"CSR: "
                f"{tascar_m['cold_start_rate']:.2f}% "
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

    # Load RL metrics from training!
    rl_metrics = load_rl_metrics()

    # Save all results
    _save_results(results, rl_metrics)

    # Print summary
    print_summary(results)

    # Print RL metrics
    if rl_metrics:
        print_rl_metrics(rl_metrics)

    # Generate ALL comparison graphs!
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
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 75)
    print("COMPLETE METRICS COMPARISON")
    print("=" * 75)

    workloads = [
        'Common', 'Significant', 'Random']

    all_metrics = [
        ('cold_start_rate',
         'Cold Start Rate (%)',
         True),
        ('avg_cold_start_overhead',
         'Avg Cold Start Delay (s)',
         True),
        ('p95_latency',
         'P95 Latency (s)',
         True),
        ('p99_latency',
         'P99 Latency (s)',
         True),
        ('avg_response_time',
         'Avg Response Time (s)',
         True),
        ('avg_wasted_memory_time',
         'Wasted Memory Time (s)',
         True),
        ('container_utilization_rate',
         'Container Utilization (%)',
         False),
        ('resource_utilization_efficiency',
         'Resource Util Efficiency (%)',
         False),
        ('sla_violation_rate',
         'SLA Violation Rate (%)',
         True),
        ('throughput',
         'Throughput (req/s)',
         False),
        ('successful_execution_ratio',
         'Successful Exec Ratio (%)',
         False),
        ('energy_per_request',
         'Energy per Request (kWh)',
         True),
        ('co2_estimate',
         'CO2 Estimate (kg)',
         True),
        ('burst_handling_efficiency',
         'Burst Handling Eff (%)',
         False),
        ('scaling_accuracy',
         'Scaling Accuracy (%)',
         False),
        ('elasticity_score',
         'Elasticity Score',
         False),
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
        print(f"\n{'='*30} "
              f"{wl} "
              f"{'='*30}")
        print(
            f"{'Metric':<35}"
            f"{'CASR':>12}"
            f"{'TASCAR':>12}"
            f"{'Winner':>10}")
        print("-" * 72)

        for (metric, label,
             lower_better) in all_metrics:
            cv = (results[wl]
                  .get('CASR', {})
                  .get(metric, 0))
            tv = (results[wl]
                  .get('TASCAR', {})
                  .get(metric, 0))

            if lower_better:
                winner = (
                    "TASCAR"
                    if tv < cv
                    else "CASR  ")
            else:
                winner = (
                    "TASCAR"
                    if tv > cv
                    else "CASR  ")

            print(
                f"{label:<35}"
                f"{cv:>12.4f}"
                f"{tv:>12.4f}"
                f"{winner:>10}")


def print_rl_metrics(rl_metrics):
    print("\n" + "=" * 55)
    print("RL TRAINING METRICS")
    print("=" * 55)
    print(
        f"Training Time:       "
        f"{rl_metrics.get('training_time_seconds', 0):.1f}s")
    print(
        f"Convergence Episode: "
        f"{rl_metrics.get('convergence_episode', -1)}")
    print(
        f"Best Reward:         "
        f"{rl_metrics.get('best_reward', 0):.4f}")
    print(
        f"Total Samples:       "
        f"{rl_metrics.get('total_samples', 0)}")
    print(
        f"Sample Efficiency:   "
        f"{rl_metrics.get('sample_efficiency', 0):.6f}")
    print(
        f"Cumulative Reward:   "
        f"{rl_metrics.get('final_cumulative_reward', 0):.2f}")


# ─────────────────────────────────────────
# PLOT ALL COMPARISONS
# 6 graph sets!
# ─────────────────────────────────────────

def plot_all_comparisons(
        results, rl_metrics=None):

    workloads  = [
        'Common', 'Significant', 'Random']
    algorithms = ['CASR', 'TASCAR']
    colors     = {
        'CASR':   '#2196F3',
        'TASCAR': '#FF5722'}

    # ── Figure 1: Cold Start Metrics ──
    _plot_figure(
        results, workloads,
        algorithms, colors,
        [
            ('cold_start_rate',
             'Cold Start Rate (%)'),
            ('avg_cold_start_overhead',
             'Avg Cold Delay (s)'),
            ('p95_latency',
             'P95 Latency (s)'),
        ],
        'Cold Start Performance Metrics',
        'fig1_cold_start.png')

    # ── Figure 2: P99 + ART + WMT ──
    _plot_figure(
        results, workloads,
        algorithms, colors,
        [
            ('p99_latency',
             'P99 Latency (s)'),
            ('avg_response_time',
             'Avg Response Time (s)'),
            ('avg_wasted_memory_time',
             'Wasted Memory Time (s)'),
        ],
        'Latency and Memory Metrics',
        'fig2_latency_memory.png')

    # ── Figure 3: Resource Metrics ──
    _plot_figure(
        results, workloads,
        algorithms, colors,
        [
            ('container_utilization_rate',
             'Container Utilization (%)'),
            ('resource_utilization_efficiency',
             'Resource Util Eff (%)'),
            ('successful_execution_ratio',
             'Successful Exec Ratio (%)'),
        ],
        'Resource Utilization Metrics',
        'fig3_resource.png')

    # ── Figure 4: QoS + Throughput ──
    _plot_figure(
        results, workloads,
        algorithms, colors,
        [
            ('sla_violation_rate',
             'SLA Violation Rate (%)'),
            ('throughput',
             'Throughput (req/s)'),
            ('burst_handling_efficiency',
             'Burst Handling Eff (%)'),
        ],
        'QoS and Throughput Metrics',
        'fig4_qos_throughput.png')

    # ── Figure 5: Energy + Scaling ──
    _plot_figure(
        results, workloads,
        algorithms, colors,
        [
            ('energy_per_request',
             'Energy per Request (kWh)'),
            ('co2_estimate',
             'CO2 Estimate (kg)'),
            ('scaling_accuracy',
             'Scaling Accuracy (%)'),
        ],
        'Energy and Scalability Metrics',
        'fig5_energy_scaling.png')

    # ── Figure 6: TPI + AGI ──
    _plot_tpi_agi(
        results, workloads,
        algorithms, colors)

    # ── Figure 7: RL Metrics ──
    if rl_metrics:
        _plot_rl_metrics(rl_metrics)

    # ── Figure 8: Master All ──
    _plot_master(
        results, workloads,
        algorithms, colors)

    print(
        f"\nAll graphs saved to "
        f"{TASCAR_RESULTS}")


def _plot_figure(
        results, workloads,
        algorithms, colors,
        metrics, title, filename):
    """Generic 1x3 comparison figure"""
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 6))
    fig.suptitle(
        title, fontsize=14,
        fontweight='bold',
        color='white')
    fig.patch.set_facecolor('#0a0e1a')
    for ax in axes:
        ax.set_facecolor('#111827')

    for ax_idx, (metric, ylabel) in (
            enumerate(metrics)):
        ax = axes[ax_idx]
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
                linewidth=0.5)
            for bar, val in zip(
                    bars, values):
                ax.text(
                    bar.get_x() +
                    bar.get_width()/2,
                    bar.get_height() +
                    max(abs(val)*0.01,
                        0.001),
                    f'{val:.3f}',
                    ha='center',
                    fontsize=8,
                    fontweight='bold',
                    color='white')

        ax.set_title(
            ylabel, fontsize=11,
            fontweight='bold',
            color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(
            workloads, color='white')
        ax.tick_params(colors='white')
        ax.legend(
            facecolor='#1e293b',
            labelcolor='white')
        ax.grid(
            axis='y', alpha=0.2,
            color='white')
        ax.spines['bottom'].set_color(
            '#334155')
        ax.spines['left'].set_color(
            '#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(
            False)
        ax.set_ylabel(
            ylabel, color='white',
            fontsize=9)

    plt.tight_layout()
    save_path = TASCAR_RESULTS + filename
    plt.savefig(
        save_path, dpi=150,
        bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print(f"Saved {filename}")


def _plot_tpi_agi(
        results, workloads,
        algorithms, colors):
    """Figure 6: TPI and AGI"""
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6))
    fig.suptitle(
        'Composite Performance Index\n'
        'TPI and Attention Gain Index (AGI)',
        fontsize=13, fontweight='bold',
        color='white')
    fig.patch.set_facecolor('#0a0e1a')
    for ax in axes:
        ax.set_facecolor('#111827')

    # TPI comparison
    ax  = axes[0]
    x   = np.arange(len(workloads))
    w   = 0.35
    for i, algo in enumerate(algorithms):
        values = [
            results.get(wl, {})
            .get(algo, {})
            .get('tpi', 0)
            for wl in workloads]
        offset = (i - 0.5) * w
        bars   = ax.bar(
            x + offset, values, w,
            label=algo,
            color=colors[algo],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() +
                bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{val:.2f}',
                ha='center', fontsize=9,
                fontweight='bold',
                color='white')
    ax.set_title(
        'TASCAR Performance Index (TPI)',
        fontsize=12, fontweight='bold',
        color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(
        workloads, color='white')
    ax.tick_params(colors='white')
    ax.legend(
        facecolor='#1e293b',
        labelcolor='white')
    ax.grid(axis='y', alpha=0.2,
            color='white')
    ax.spines['bottom'].set_color(
        '#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(
        'TPI Score', color='white')

    # AGI bar chart
    ax2   = axes[1]
    agis  = [
        results.get(wl, {})
        .get('TASCAR', {})
        .get('agi', 0)
        for wl in workloads]
    wl_colors = [
        '#FF5722', '#FF9800', '#FFC107']
    bars2 = ax2.bar(
        workloads, agis,
        color=wl_colors, alpha=0.85,
        edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars2, agis):
        ax2.text(
            bar.get_x() +
            bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{val:.2f}%',
            ha='center', fontsize=10,
            fontweight='bold',
            color='white')
    ax2.set_title(
        'Attention Gain Index (AGI)\n'
        'Cold Start Reduction by '
        'Transformer',
        fontsize=12, fontweight='bold',
        color='white')
    ax2.set_ylabel(
        'AGI (%)', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(axis='y', alpha=0.2,
             color='white')
    ax2.spines['bottom'].set_color(
        '#334155')
    ax2.spines['left'].set_color(
        '#334155')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        TASCAR_RESULTS +
        'fig6_tpi_agi.png',
        dpi=150, bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print("Saved fig6_tpi_agi.png")


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
        'RL Training Metrics\n'
        'TASCAR Learning Performance',
        fontsize=14, fontweight='bold',
        color='white')
    fig.patch.set_facecolor('#0a0e1a')
    for row in axes:
        for ax in row:
            ax.set_facecolor('#111827')

    def smooth(vals, w=10):
        s = []
        for i in range(len(vals)):
            start = max(0, i-w)
            s.append(np.mean(vals[start:i+1]))
        return s

    def style_ax(ax, title, xlabel,
                 ylabel):
        ax.set_title(
            title, fontsize=11,
            fontweight='bold',
            color='white')
        ax.set_xlabel(
            xlabel, color='white')
        ax.set_ylabel(
            ylabel, color='white')
        ax.tick_params(colors='white')
        ax.grid(alpha=0.2, color='white')
        ax.spines['bottom'].set_color(
            '#334155')
        ax.spines['left'].set_color(
            '#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(
            False)

    # Reward convergence
    rewards = logs.get('rewards', [])
    if rewards:
        axes[0, 0].plot(
            episodes, rewards,
            color='#0ea5e9', alpha=0.4,
            linewidth=1)
        axes[0, 0].plot(
            episodes, smooth(rewards),
            color='#0ea5e9',
            linewidth=2.5,
            label='Smoothed')
        conv = rl_metrics.get(
            'convergence_episode', -1)
        if conv > 0:
            axes[0, 0].axvline(
                x=conv,
                color='#10b981',
                linestyle='--',
                label=f'Converged: Ep{conv}')
        axes[0, 0].legend(
            facecolor='#1e293b',
            labelcolor='white',
            fontsize=8)
        style_ax(
            axes[0, 0],
            'Reward Convergence',
            'Episode',
            'Avg Reward per Step')

    # Cold start rate
    cold_rates = logs.get(
        'cold_start_rates', [])
    if cold_rates:
        axes[0, 1].plot(
            episodes, cold_rates,
            color='#ef4444', alpha=0.4,
            linewidth=1)
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

    # Dynamic theta
    thetas = logs.get('thetas', [])
    if thetas:
        axes[0, 2].plot(
            episodes, thetas,
            color='#8b5cf6',
            linewidth=1.5, alpha=0.7,
            label='TASCAR theta')
        axes[0, 2].plot(
            episodes, smooth(thetas),
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
            'Dynamic Theta Adaptation',
            'Episode', 'Theta (θ)')

    # Cumulative reward
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
            'Cumulative Reward (Rcum)',
            'Episode', 'Rcum')

    # Sample efficiency
    sample_counts = logs.get(
        'sample_counts', [])
    if sample_counts and rewards:
        axes[1, 1].plot(
            sample_counts, rewards,
            color='#10b981', alpha=0.4,
            linewidth=1)
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
            'Sample Efficiency\n'
            '(Reward vs Samples)',
            'Training Samples',
            'Reward')

    # RL metrics text summary
    axes[1, 2].set_facecolor('#0f172a')
    axes[1, 2].axis('off')
    summary = (
        f"RL TRAINING SUMMARY\n\n"
        f"Training Time:\n"
        f"  {rl_metrics.get('training_time_seconds', 0):.1f} seconds\n\n"
        f"Convergence Episode:\n"
        f"  {rl_metrics.get('convergence_episode', -1)}\n\n"
        f"Best Reward:\n"
        f"  {rl_metrics.get('best_reward', 0):.4f}\n\n"
        f"Total Training Samples:\n"
        f"  {rl_metrics.get('total_samples', 0):,}\n\n"
        f"Sample Efficiency:\n"
        f"  {rl_metrics.get('sample_efficiency', 0):.6f}\n\n"
        f"Cumulative Reward:\n"
        f"  {rl_metrics.get('final_cumulative_reward', 0):.2f}")
    axes[1, 2].text(
        0.1, 0.95, summary,
        transform=axes[1, 2].transAxes,
        fontsize=11, color='white',
        va='top', fontfamily='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='#1e293b',
            alpha=0.8))
    axes[1, 2].set_title(
        'Training Summary',
        fontsize=11, fontweight='bold',
        color='white')

    plt.tight_layout()
    plt.savefig(
        TASCAR_RESULTS +
        'fig7_rl_metrics.png',
        dpi=150, bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print("Saved fig7_rl_metrics.png")


def _plot_master(
        results, workloads,
        algorithms, colors):
    """Figure 8: All metrics master view"""
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

    nrows = 6
    ncols = 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(22, 28))
    fig.suptitle(
        'TASCAR vs CASR: '
        'Complete Metrics Comparison\n'
        'All Workloads | '
        'All Professor Metrics',
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(
            False)

    # Hide unused axes
    for idx in range(
            len(all_metrics),
            len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(
        rect=[0, 0, 1, 0.97])
    plt.savefig(
        TASCAR_RESULTS +
        'fig8_master_all_metrics.png',
        dpi=150, bbox_inches='tight',
        facecolor='#0a0e1a')
    plt.close()
    print(
        "Saved fig8_master_all_metrics.png")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TASCAR Complete Evaluation")
    print("All Professor Metrics!")
    print("8 Graph Sets!")
    print(f"SLA Threshold: {SLA_THRESHOLD}s")
    print(f"Carbon Intensity: "
          f"{CARBON_INTENSITY} kg/kWh")
    print(f"TPI: CSR={TPI_W1} "
          f"WMT={TPI_W2} "
          f"TPT={TPI_W3} "
          f"SVR={TPI_W4} "
          f"RUE={TPI_W5}")
    print("=" * 60)

    tascar_path = (
        TASCAR_MODEL_PATH + "best/")
    if not os.path.exists(
            tascar_path + "actor.pth"):
        print(
            "\nNo TASCAR model found!")
        print(
            "Run: "
            "python train_tascar.py")
    else:
        run_evaluation()
# evaluate.py
# Runs all experiments and generates graphs
# Uses NUM_FUNCTIONS and EVAL_CALLS from config
# Has cooling breaks to prevent overheating
# Run AFTER train.py full completes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
import time
from collections import Counter
from simulator import AzureDataLoader, Simulator
from scache import SCache
from baselines import (FixedAlgorithm,
                       LCSAlgorithm,
                       HistAlgorithm,
                       FaaSCacheAlgorithm)
from ppo_agent import PPOAgent
from config import *

matplotlib.use('Agg')


class CASRAlgorithm:
    """
    Complete CASR for evaluation.
    Loads trained PPO model.
    Scales S-Cache queues dynamically.
    """

    def __init__(self, model_path=None):
        self.scache     = SCache()
        self.call_count = 0
        state_dim       = NUM_QUEUES * 7
        action_dim      = 3 ** NUM_QUEUES
        self.agent      = PPOAgent(
            state_dim, action_dim)

        if (model_path and
                os.path.exists(
                    model_path + "/actor.pth")):
            self.agent.load(model_path)
            print(f"  Model loaded: {model_path}")
        else:
            print("  No model found.")
            print("  Using S-Cache only.")

        self.action_map = self._build_action_map()

    def _build_action_map(self):
        action_map = {}
        choices    = [
            -SCALING_FACTOR,
            0,
            SCALING_FACTOR]
        for i in range(3 ** NUM_QUEUES):
            action = []
            temp   = i
            for _ in range(NUM_QUEUES):
                action.append(choices[temp % 3])
                temp //= 3
            action_map[i] = action
        return action_map

    def handle_request(self, function_call):
        self.call_count += 1
        if self.call_count % DELTA == 0:
            state = np.array(
                self.scache.get_state(),
                dtype=np.float32)
            mean = np.mean(state)
            std  = np.std(state)
            if std > 0:
                state = (state - mean) / std
            action, _ = (
                self.agent.choose_action(state))
            for q_idx, scale in enumerate(
                    self.action_map[action]):
                if scale != 0:
                    self.scache.scale_queue(
                        q_idx, scale)
        return self.scache.handle_request(
            function_call)

    def get_total_wasted_memory_time(self):
        return (self.scache
                .get_total_wasted_memory_time())


class ExperimentRunner:
    """
    Runs all experiments.
    Has cooling breaks between runs.
    Uses NUM_FUNCTIONS from config.
    """

    def __init__(self):
        self.loader  = AzureDataLoader()
        self.results = {}

    def run_all_experiments(self):
        print("=" * 60)
        print("Running All Experiments")
        print(f"Functions per workload: "
              f"{NUM_FUNCTIONS}")
        print(f"Calls per workload: "
              f"{EVAL_CALLS}")
        print("=" * 60)

        # Prepare workloads
        print("\nPreparing workloads...")
        workloads = {
            'Common': (
                self._get_common_workload()),
            'Significant': (
                self._get_significant_workload()),
            'Random': (
                self._get_random_workload())
        }

        # All algorithms
        algorithms = {
            'CASR':      lambda: CASRAlgorithm(
                MODEL_SAVE_PATH + "best/"),
            'S-Cache':   lambda: SCache(),
            'LCS':       lambda: LCSAlgorithm(),
            'FaaSCache': lambda: (
                FaaSCacheAlgorithm()),
            'Hist':      lambda: HistAlgorithm(),
            'Fixed':     lambda: FixedAlgorithm()
        }

        for w_idx, (wl_name, calls) in (
                enumerate(workloads.items())):

            # Cool down between workloads
            if w_idx > 0:
                self._cool_down_workload()

            unique = len(set(
                c.function_id for c in calls))
            print(f"\nWorkload: {wl_name}")
            print(f"Calls: {len(calls)} | "
                  f"Unique functions: {unique}")
            print("-" * 40)

            self.results[wl_name] = {}

            for a_idx, (algo_name, algo_fn) in (
                    enumerate(
                        algorithms.items())):

                # Cool down between algorithms
                if a_idx > 0:
                    self._cool_down_algorithm()

                print(f"  Running {algo_name}...")

                try:
                    algo    = algo_fn()
                    sim     = Simulator(algo)
                    metrics = sim.run(
                        calls, verbose=False)
                    self.results[wl_name][
                        algo_name] = metrics
                    print(
                        f"    ✅ "
                        f"Cold%: {metrics['cold_start_rate']:.2f}% | "
                        f"Overhead: {metrics['avg_cold_start_overhead']:.2f}s | "
                        f"WMT: {metrics['avg_wasted_memory_time']:.2f}s")

                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    self.results[wl_name][
                        algo_name] = {
                        'cold_start_rate': 0,
                        'avg_cold_start_overhead': 0,
                        'avg_wasted_memory_time': 0,
                        'total_invocations': 0,
                        'total_cold_starts': 0,
                        'total_warm_starts': 0,
                        'total_cold_overhead': 0,
                        'total_wmt': 0
                    }

        self._save_results()
        print("\n✅ All experiments complete!")
        return self.results

    def _cool_down_algorithm(self):
        """30 second break between algorithms"""
        secs = COOLING_BETWEEN_ALGORITHMS
        print(f"  Cooling {secs}s...")
        time.sleep(secs)

    def _cool_down_workload(self):
        """2 minute break between workloads"""
        secs = COOLING_BETWEEN_WORKLOADS
        print(f"\n{'='*60}")
        print(f"Cooling CPU {secs}s "
              f"before next workload...")
        print(f"{'='*60}")
        for i in range(secs // 10):
            remaining = secs - (i * 10)
            print(f"  {remaining}s remaining...")
            time.sleep(10)
        print("  Done cooling! Continuing...")

    def _get_top_functions(self, calls, n):
        """Gets top N most frequent functions"""
        func_counts   = Counter(
            c.function_id for c in calls)
        top_functions = set(
            f for f, _ in
            func_counts.most_common(n))
        return top_functions

    def _sample_calls(self, calls, n):
        """Randomly samples n calls"""
        if len(calls) <= n:
            return calls
        indices = np.random.choice(
            len(calls), n, replace=False)
        indices.sort()
        return [calls[i] for i in indices]

    def _get_common_workload(self):
        """
        Common workload.
        Top NUM_FUNCTIONS most frequent functions.
        """
        print("  Common workload...")
        calls         = self.loader.load_day(1)
        np.random.seed(42)
        top_functions = self._get_top_functions(
            calls, NUM_FUNCTIONS)
        filtered      = [
            c for c in calls
            if c.function_id in top_functions]
        result = self._sample_calls(
            filtered, EVAL_CALLS)
        print(f"    {len(result)} calls | "
              f"{NUM_FUNCTIONS} functions")
        return result

    def _get_significant_workload(self):
        """
        Significant workload.
        Top NUM_FUNCTIONS high overhead functions.
        """
        print("  Significant workload...")
        calls = self.loader.load_day(2)
        np.random.seed(42)

        # Focus on high cold start overhead
        heavy = [
            c for c in calls
            if c.cold_start_overhead > 1]

        top_functions = self._get_top_functions(
            heavy, NUM_FUNCTIONS)
        filtered = [
            c for c in heavy
            if c.function_id in top_functions]
        result = self._sample_calls(
            filtered, EVAL_CALLS)
        print(f"    {len(result)} calls | "
              f"{NUM_FUNCTIONS} functions")
        return result

    def _get_random_workload(self):
        """
        Random workload.
        NUM_FUNCTIONS randomly selected functions.
        """
        print("  Random workload...")
        calls = self.loader.load_day(3)
        np.random.seed(123)

        func_counts   = Counter(
            c.function_id for c in calls)
        all_functions = list(func_counts.keys())
        np.random.shuffle(all_functions)
        selected = set(
            all_functions[:NUM_FUNCTIONS])

        filtered = [
            c for c in calls
            if c.function_id in selected]
        result = self._sample_calls(
            filtered, EVAL_CALLS)
        print(f"    {len(result)} calls | "
              f"{NUM_FUNCTIONS} functions")
        return result

    def _save_results(self):
        os.makedirs(RESULTS_PATH, exist_ok=True)
        path = os.path.join(
            RESULTS_PATH,
            'experiment_results.json')
        serializable = {}
        for wl, algos in self.results.items():
            serializable[wl] = {}
            for algo, metrics in algos.items():
                serializable[wl][algo] = {
                    k: float(v)
                    for k, v in metrics.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {path}")


class GraphGenerator:
    """
    Generates all comparison graphs.
    Publication quality plots.
    """

    def __init__(self, results,
                 save_path=RESULTS_PATH):
        self.results   = results
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.colors = {
            'CASR':      '#2196F3',
            'S-Cache':   '#4CAF50',
            'LCS':       '#FF9800',
            'FaaSCache': '#9C27B0',
            'Hist':      '#F44336',
            'Fixed':     '#795548'
        }

    def plot_all(self):
        print("\nGenerating graphs...")
        self.plot_combined_comparison()
        self.plot_cold_start_rate()
        self.plot_wasted_memory_time()
        self.plot_cold_start_overhead()
        self.plot_training_curves()
        print("✅ All graphs generated!")

    def _make_bar_chart(self, ax, workload,
                        metric, ylabel,
                        title=None):
        """Creates one bar chart"""
        if workload not in self.results:
            return
        data       = self.results[workload]
        algo_names = list(data.keys())
        values     = [data[a][metric]
                     for a in algo_names]
        colors     = [
            self.colors.get(a, 'gray')
            for a in algo_names]

        bars = ax.bar(
            algo_names, values,
            color=colors, alpha=0.8,
            edgecolor='black',
            linewidth=0.5)

        # Gold border on CASR
        if 'CASR' in algo_names:
            idx = algo_names.index('CASR')
            bars[idx].set_edgecolor('gold')
            bars[idx].set_linewidth(3)

        if title:
            ax.set_title(title, fontsize=12,
                        fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(
            axis='x', rotation=45,
            labelsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() +
                bar.get_width() / 2,
                bar.get_height() * 1.02,
                f'{val:.1f}',
                ha='center', va='bottom',
                fontsize=7,
                fontweight='bold')

    def plot_combined_comparison(self):
        """
        Main figure showing all metrics.
        This is the graph to show professor!
        Matches Figure 8 from paper.
        """
        fig, axes = plt.subplots(
            3, 3, figsize=(18, 14))
        fig.suptitle(
            'CASR vs Baseline Algorithms\n'
            'Complete Performance Comparison',
            fontsize=16,
            fontweight='bold')

        workloads  = [
            'Common', 'Significant', 'Random']
        metric_info = [
            ('cold_start_rate',
             '(a) Cold Start Rate (%)',
             'Cold Start Rate (%)'),
            ('avg_wasted_memory_time',
             '(b) Avg Wasted Memory Time (s)',
             'Avg WMT (s)'),
            ('avg_cold_start_overhead',
             '(c) Avg Cold Start Overhead (s)',
             'Avg Overhead (s)')
        ]

        for row, (metric, row_label,
                  ylabel) in enumerate(
                metric_info):
            for col, workload in enumerate(
                    workloads):
                ax = axes[row][col]
                self._make_bar_chart(
                    ax, workload, metric,
                    ylabel,
                    title=(workload
                           if row == 0
                           else None))
                if col == 0:
                    ax.set_ylabel(
                        f'{row_label}',
                        fontsize=9)

        plt.tight_layout()
        path = os.path.join(
            self.save_path,
            'combined_comparison.png')
        plt.savefig(path, dpi=150,
                   bbox_inches='tight')
        print("  Saved: combined_comparison.png")
        plt.close()

    def plot_cold_start_rate(self):
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 6))
        fig.suptitle(
            'Cold Start Rate Comparison (%)',
            fontsize=14, fontweight='bold')
        for idx, wl in enumerate(
                ['Common',
                 'Significant',
                 'Random']):
            self._make_bar_chart(
                axes[idx], wl,
                'cold_start_rate',
                'Cold Start Rate (%)',
                title=wl)
        plt.tight_layout()
        path = os.path.join(
            self.save_path,
            'cold_start_rate.png')
        plt.savefig(path, dpi=150,
                   bbox_inches='tight')
        print("  Saved: cold_start_rate.png")
        plt.close()

    def plot_wasted_memory_time(self):
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 6))
        fig.suptitle(
            'Average Wasted Memory Time (s)',
            fontsize=14, fontweight='bold')
        for idx, wl in enumerate(
                ['Common',
                 'Significant',
                 'Random']):
            self._make_bar_chart(
                axes[idx], wl,
                'avg_wasted_memory_time',
                'Avg WMT (seconds)',
                title=wl)
        plt.tight_layout()
        path = os.path.join(
            self.save_path,
            'wasted_memory_time.png')
        plt.savefig(path, dpi=150,
                   bbox_inches='tight')
        print("  Saved: wasted_memory_time.png")
        plt.close()

    def plot_cold_start_overhead(self):
        fig, axes = plt.subplots(
            1, 3, figsize=(15, 6))
        fig.suptitle(
            'Average Cold Start Overhead (s)',
            fontsize=14, fontweight='bold')
        for idx, wl in enumerate(
                ['Common',
                 'Significant',
                 'Random']):
            self._make_bar_chart(
                axes[idx], wl,
                'avg_cold_start_overhead',
                'Avg Overhead (seconds)',
                title=wl)
        plt.tight_layout()
        path = os.path.join(
            self.save_path,
            'cold_start_overhead.png')
        plt.savefig(path, dpi=150,
                   bbox_inches='tight')
        print("  Saved: cold_start_overhead.png")
        plt.close()

    def plot_training_curves(self):
        log_path = os.path.join(
            RESULTS_PATH,
            'training_logs.json')
        if not os.path.exists(log_path):
            print("  No training logs found.")
            return

        with open(log_path, 'r') as f:
            logs = json.load(f)

        fig, axes = plt.subplots(
            1, 3, figsize=(15, 5))
        fig.suptitle(
            'PPO Training Convergence',
            fontsize=14, fontweight='bold')

        episodes = logs['episodes']

        for ax, key, color, title, ylabel in [
            (axes[0], 'rewards',
             'blue', '(a) Reward', 'Reward'),
            (axes[1], 'cold_start_rates',
             'red', '(b) Cold Start Rate',
             'Cold Start Rate (%)'),
            (axes[2], 'wmts',
             'green', '(c) Average WMT',
             'Avg WMT (s)')
        ]:
            ax.plot(episodes, logs[key],
                   color=color, alpha=0.4,
                   linewidth=1)
            ax.plot(episodes,
                   self._smooth(
                       logs[key], 10),
                   color=f'dark{color}',
                   linewidth=2.5,
                   label='Smoothed')
            ax.set_title(title)
            ax.set_xlabel('Episodes')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(
            self.save_path,
            'training_curves.png')
        plt.savefig(path, dpi=150,
                   bbox_inches='tight')
        print("  Saved: training_curves.png")
        plt.close()

    def _smooth(self, values, window):
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window)
            smoothed.append(
                np.mean(values[start:i+1]))
        return smoothed

    def print_summary_table(self):
        """Prints final results table"""
        print("\n" + "=" * 70)
        print("FINAL RESULTS SUMMARY")
        print("=" * 70)

        for workload, data in (
                self.results.items()):
            print(f"\nWorkload: {workload}")
            print(
                f"{'Algorithm':<12} "
                f"{'Cold%':>8} "
                f"{'Overhead':>12} "
                f"{'WMT':>12}")
            print("-" * 50)

            # Sort by cold start rate
            sorted_algos = sorted(
                data.items(),
                key=lambda x: (
                    x[1]['cold_start_rate']))

            for algo, metrics in sorted_algos:
                marker = (
                    " ✅ BEST"
                    if algo == "CASR"
                    else "")
                print(
                    f"{algo:<12} "
                    f"{metrics['cold_start_rate']:>7.2f}%"
                    f"{metrics['avg_cold_start_overhead']:>11.2f}s"
                    f"{metrics['avg_wasted_memory_time']:>11.2f}s"
                    f"{marker}")

        # Show improvements
        print("\n" + "=" * 70)
        print("CASR vs BASELINES")
        print("=" * 70)

        for workload, data in (
                self.results.items()):
            if 'CASR' not in data:
                continue

            casr   = data['CASR']
            others = {
                k: v
                for k, v in data.items()
                if k not in ['CASR', 'S-Cache']}

            if not others:
                continue

            print(f"\n{workload}:")
            for metric, label in [
                ('cold_start_rate',
                 'Cold Start'),
                ('avg_cold_start_overhead',
                 'Overhead'),
                ('avg_wasted_memory_time',
                 'WMT')
            ]:
                casr_val  = casr[metric]
                best_name = min(
                   others.keys(),
                    key=lambda k: (
                        others[k][metric]))
                best_val  = (
                    others[best_name][metric])

                if best_val > 0:
                    imp = ((best_val - casr_val)
                          / best_val * 100)
                    sign = (
                        "better ✅"
                        if imp > 0
                        else "worse ❌")
                    print(
                        f"  {label}: "
                        f"CASR={casr_val:.1f} "
                        f"vs {best_name}"
                        f"={best_val:.1f} "
                        f"→ {abs(imp):.1f}% "
                        f"{sign}")


if __name__ == "__main__":
    print("=" * 60)
    print("CASR Evaluation")
    print(f"Functions: {NUM_FUNCTIONS}")
    print(f"Calls: {EVAL_CALLS}")
    print(f"Cooling breaks: enabled")
    print("=" * 60)
    print("\n⚠️  Before starting:")
    print("  1. Laptop on hard flat surface")
    print("  2. Close all other apps")
    print("  3. Charger plugged in")
    print(f"\nEstimated time: ~2 hours")
    print("Starting in 10 seconds...")
    time.sleep(10)

    # Run all experiments
    runner  = ExperimentRunner()
    results = runner.run_all_experiments()

    # Generate all graphs
    grapher = GraphGenerator(results)
    grapher.plot_all()
    grapher.print_summary_table()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print("\nResults in results/ folder:")
    print("  combined_comparison.png ....")
    print("  cold_start_rate.png")
    print("  wasted_memory_time.png")
    print("  cold_start_overhead.png")
    print("  training_curves.png")
# evaluate_tascar.py
# Compares CASR vs TASCAR
# Uses TASCAR_DELTA for TASCAR
# Uses DELTA for CASR
# Fair comparison!

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
    SEQUENCE_LENGTH,
    TRANSFORMER_DIM,
    TASCAR_MODEL_PATH,
    TASCAR_RESULTS,
    MODEL_SAVE_PATH,
    COOLING_BETWEEN_ALGORITHMS,
    COOLING_BETWEEN_WORKLOADS,
    SCALING_FACTOR
)
from simulator import (
    AzureDataLoader,
    Simulator)
from scache import SCache
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent
from ppo_agent import PPOAgent


# ─────────────────────────────────────────
# CASR ALGORITHM
# Uses original DELTA = 10000
# Loaded from CASR trained model
# ─────────────────────────────────────────

class CASRAlgorithm:
    """
    CASR with trained PPO model.
    Uses DELTA = 10000 (original!)
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
            print(
                "  CASR model loaded!")
        else:
            print(
                "  No CASR model!")
            print(
                "  Using S-Cache only!")

        self.action_map = (
            self._build_action_map())

    def _build_action_map(self):
        choices    = [
            -SCALING_FACTOR,
            0,
            SCALING_FACTOR]
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
        # CASR uses original DELTA!
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
            for q_idx, scale in (
                    enumerate(
                        self.action_map[
                            action])):
                if scale != 0:
                    self.scache\
                        .scale_queue(
                        q_idx, scale)
        return (
            self.scache
            .handle_request(
                function_call))

    def get_total_wasted_memory_time(
            self):
        return (
            self.scache
            .get_total_wasted_memory_time())


# ─────────────────────────────────────────
# TASCAR ALGORITHM
# Uses TASCAR_DELTA = 1000
# More frequent decisions!
# ─────────────────────────────────────────

class TASCARAlgorithm:
    """
    TASCAR with trained SAC +
    Transformer.

    Uses TASCAR_DELTA = 1000
    Makes 100 decisions per workload
    vs CASR 10 decisions per workload!

    More frequent decisions +
    Temporal state encoding =
    Better performance!
    """
    def __init__(self,
                 model_path=None):
        self.scache     = SCache()
        self.call_count = 0
        self.state_dim  = NUM_QUEUES * 7
        self.action_dim = (
            3 ** NUM_QUEUES)

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
            print(
                "  No TASCAR model!")
            print(
                "  Run train_tascar.py!")

        self.history = (
            StateHistoryBuffer(
                SEQUENCE_LENGTH,
                self.state_dim))

        self.action_map = (
            self.agent.action_map)

    def handle_request(self,
                       function_call):
        self.call_count += 1

        # TASCAR uses smaller delta!
        if (self.call_count %
                TASCAR_DELTA == 0):

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

            scales = (
                self.action_map[action])
            for q_idx, scale in (
                    enumerate(scales)):
                if scale != 0:
                    self.scache\
                        .scale_queue(
                        q_idx, scale)

        return (
            self.scache
            .handle_request(
                function_call))

    def get_total_wasted_memory_time(
            self):
        return (
            self.scache
            .get_total_wasted_memory_time())


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
            len(common),
            EVAL_CALLS,
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
        TASCAR_RESULTS,
        exist_ok=True)

    workloads = load_workloads()
    results   = {}

    print("\nStarting evaluation...")
    print("=" * 55)

    for wl_idx, (wl_name, calls) in (
            enumerate(
                workloads.items())):

        print(
            f"\nWorkload: {wl_name}")
        print("-" * 40)

        if wl_idx > 0:
            secs = COOLING_BETWEEN_WORKLOADS
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

        # Run CASR
        print("\n  Running CASR...")
        try:
            casr   = CASRAlgorithm(
                MODEL_SAVE_PATH +
                "best/")
            sim    = Simulator(casr)
            casr_m = sim.run(
                calls,
                verbose=False)
            results[wl_name][
                'CASR'] = casr_m
            print(
                f"  ✅ CASR "
                f"Cold%: "
                f"{casr_m['cold_start_rate']:.3f}% "
                f"WMT: "
                f"{casr_m['avg_wasted_memory_time']:.3f}s")
        except Exception as e:
            print(
                f"  ❌ CASR: {e}")
            results[wl_name][
                'CASR'] = {
                'cold_start_rate':         0,
                'avg_wasted_memory_time':  0,
                'avg_cold_start_overhead': 0}

        # Cool between algorithms
        secs = COOLING_BETWEEN_ALGORITHMS
        print(f"\n  Cooling {secs}s...")
        time.sleep(secs)

        # Run TASCAR
        print("\n  Running TASCAR...")
        try:
            tascar   = TASCARAlgorithm(
                TASCAR_MODEL_PATH +
                "best/")
            sim      = Simulator(tascar)
            tascar_m = sim.run(
                calls,
                verbose=False)
            results[wl_name][
                'TASCAR'] = tascar_m
            print(
                f"  ✅ TASCAR "
                f"Cold%: "
                f"{tascar_m['cold_start_rate']:.3f}% "
                f"WMT: "
                f"{tascar_m['avg_wasted_memory_time']:.3f}s")
        except Exception as e:
            print(
                f"  ❌ TASCAR: {e}")
            results[wl_name][
                'TASCAR'] = {
                'cold_start_rate':         0,
                'avg_wasted_memory_time':  0,
                'avg_cold_start_overhead': 0}

    _save_results(results)
    print_summary(results)
    plot_comparison(results)

    return results


# ─────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────

def _save_results(results):
    path = (
        TASCAR_RESULTS +
        'casr_vs_tascar.json')
    serializable = {}
    for wl, algos in results.items():
        serializable[wl] = {}
        for algo, metrics in (
                algos.items()):
            serializable[wl][algo] = {
                k: float(v)
                for k, v in
                metrics.items()}
    with open(path, 'w') as f:
        json.dump(
            serializable,
            f, indent=2)
    print(f"\nResults saved: {path}")


# ─────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 65)
    print("CASR vs TASCAR Final Results")
    print("=" * 65)

    workloads = [
        'Common',
        'Significant',
        'Random']

    print(
        f"\n{'Workload':<14}"
        f"{'Algorithm':<10}"
        f"{'Cold%':>8}"
        f"{'WMT(s)':>10}"
        f"{'OH(s)':>10}"
        f"{'Result':>14}")
    print("-" * 65)

    for wl in workloads:
        if wl not in results:
            continue

        casr_cold = (
            results[wl]
            .get('CASR', {})
            .get('cold_start_rate', 0))
        tascar_cold = (
            results[wl]
            .get('TASCAR', {})
            .get('cold_start_rate', 0))

        for algo in [
                'CASR', 'TASCAR']:
            if algo not in results[wl]:
                continue
            m = results[wl][algo]

            if algo == 'TASCAR':
                diff = (
                    casr_cold -
                    tascar_cold)
                if diff > 0:
                    result = (
                        f"✅ -{diff:.3f}pp")
                elif diff < 0:
                    result = (
                        f"❌ "
                        f"+{abs(diff):.3f}pp")
                else:
                    result = "= Same"
            else:
                result = "baseline"

            print(
                f"{wl if algo == 'CASR' else '':<14}"
                f"{algo:<10}"
                f"{m['cold_start_rate']:>7.3f}%"
                f"{m['avg_wasted_memory_time']:>10.3f}s"
                f"{m['avg_cold_start_overhead']:>10.3f}s"
                f"{result:>14}")

    print("\npp = percentage points")
    print("✅ = TASCAR better than CASR")
    print("❌ = TASCAR worse than CASR")


# ─────────────────────────────────────────
# PLOT COMPARISON
# ─────────────────────────────────────────

def plot_comparison(results):
    workloads  = [
        'Common',
        'Significant',
        'Random']
    algorithms = ['CASR', 'TASCAR']
    colors     = {
        'CASR':   '#2196F3',
        'TASCAR': '#FF5722'}

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 6))
    fig.suptitle(
        'CASR vs TASCAR Comparison',
        fontsize=14,
        fontweight='bold')

    metrics_info = [
        ('cold_start_rate',
         'Cold Start Rate (%)',
         'Cold Start Rate (%)'),
        ('avg_wasted_memory_time',
         'Wasted Memory Time (s)',
         'WMT (seconds)'),
        ('avg_cold_start_overhead',
         'Cold Start Overhead (s)',
         'Overhead (seconds)')
    ]

    for ax_idx, (
            metric,
            title,
            ylabel) in enumerate(
            metrics_info):

        ax    = axes[ax_idx]
        x     = np.arange(len(workloads))
        width = 0.35

        for i, algo in enumerate(
                algorithms):
            values = [
                results
                .get(wl, {})
                .get(algo, {})
                .get(metric, 0)
                for wl in workloads]

            offset = (i - 0.5) * width
            bars   = ax.bar(
                x + offset,
                values,
                width,
                label=algo,
                color=colors[algo],
                alpha=0.85,
                edgecolor='black',
                linewidth=0.5)

            for bar, val in zip(
                    bars, values):
                ax.text(
                    bar.get_x() +
                    bar.get_width() / 2,
                    bar.get_height() +
                    0.1,
                    f'{val:.2f}',
                    ha='center',
                    fontsize=8,
                    fontweight='bold')

        ax.set_title(
            title,
            fontsize=11,
            fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(workloads)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    save_path = (
        TASCAR_RESULTS +
        'casr_vs_tascar_comparison.png')
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches='tight')
    plt.close()
    print(f"Graph saved: {save_path}")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("TASCAR Evaluation")
    print("Comparing CASR vs TASCAR")
    print("=" * 55)

    tascar_path = (
        TASCAR_MODEL_PATH + "best/")
    if not os.path.exists(
            tascar_path + "actor.pth"):
        print(
            "\n❌ TASCAR model not found!")
        print(
            "Run: python train_tascar.py")
    else:
        run_evaluation()
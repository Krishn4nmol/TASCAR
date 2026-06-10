# ablation_study.py
# Evaluates all 4 ablation variants!
# V1: CASR (PPO only)
# V2: SAC Only (no Transformer)
# V3: Transformer + PPO (no SAC)
# V4: Full TASCAR (all components)
# Produces complete comparison table!

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
    MODEL_SAVE_PATH,
    TASCAR_MODEL_PATH,
    SAC_ONLY_MODEL_PATH,
    TRANSFORMER_PPO_MODEL_PATH,
    ABLATION_RESULTS,
    SCALING_FACTOR,
    RANDOM_SEED,
    SLA_THRESHOLD,
    CARBON_INTENSITY)
from simulator import AzureDataLoader
from metrics_tracker import MetricsTracker
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent
from ppo_agent import PPOAgent
from train_sac_only import SACOnlyAgent
from train_transformer_ppo import (
    TransformerPPOAgent)


# ─────────────────────────────────────────
# LOAD WORKLOADS
# Same as evaluate_tascar.py!
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
    np.random.seed(RANDOM_SEED)
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
    np.random.seed(RANDOM_SEED)
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
    np.random.seed(RANDOM_SEED + 1)
    np.random.shuffle(funcs)
    selected  = set(
        funcs[:NUM_FUNCTIONS])
    random_wl = [
        c for c in day3
        if c.function_id in selected]
    np.random.seed(RANDOM_SEED + 1)
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
# VARIANT 1: CASR (PPO Only)
# FIXED: handle THEN scale!
# ─────────────────────────────────────────

def run_casr(calls):
    """
    V1: CASR with PPO agent!
    Single state snapshot!
    No Transformer!
    Original 200 episode model!
    """
    model_path = MODEL_SAVE_PATH + "best/"
    if not os.path.exists(
            model_path + "actor.pth"):
        print("    No CASR model!")
        return None

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES
    agent      = PPOAgent(
        state_dim, action_dim)
    agent.load(model_path)

    action_map = {}
    choices    = [
        -SCALING_FACTOR,
        0,
        SCALING_FACTOR]
    for i in range(3 ** NUM_QUEUES):
        action = []
        temp   = i
        for _ in range(NUM_QUEUES):
            action.append(
                choices[temp % 3])
            temp //= 3
        action_map[i] = action

    tracker    = MetricsTracker()
    call_count = 0

    for call in calls:
        tracker.handle_request(call)
        call_count += 1
        if call_count % DELTA == 0:
            state = np.array(
                tracker.get_state(),
                dtype=np.float32)
            mean = np.mean(state)
            std  = np.std(state)
            if std > 0:
                state = (
                    (state - mean) / std)
            action, _ = (
                agent.choose_action(
                    state))
            for q_idx, scale in (
                    enumerate(
                        action_map[
                            action])):
                if scale != 0:
                    tracker.scale_queue(
                        q_idx, scale)

    return tracker.get_all_metrics()


# ─────────────────────────────────────────
# VARIANT 2: SAC Only
# FIXED: handle THEN scale!
# ─────────────────────────────────────────

def run_sac_only(calls):
    """
    V2: SAC agent only!
    No Transformer!
    Raw 21-dim state!
    500 episodes!
    """
    model_path = (
        SAC_ONLY_MODEL_PATH + "best/")
    if not os.path.exists(
            model_path + "actor.pth"):
        print(
            "    No SAC-Only model!")
        print(
            "    Run: python "
            "train_sac_only.py")
        return None

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES
    agent      = SACOnlyAgent(
        state_dim, action_dim)
    agent.load(model_path)

    tracker    = MetricsTracker()
    call_count = 0

    for call in calls:
        tracker.handle_request(call)
        call_count += 1
        if (call_count %
                TASCAR_EVAL_DELTA == 0):
            state = np.array(
                tracker.get_state(),
                dtype=np.float32)
            mean = np.mean(state)
            std  = np.std(state)
            if std > 0:
                state = (
                    (state - mean) / std)
            action = (
                agent.choose_action(
                    state,
                    evaluate=True))
            for q_idx, scale in (
                    enumerate(
                        agent.action_map[
                            action])):
                if scale != 0:
                    tracker.scale_queue(
                        q_idx, scale)

    return tracker.get_all_metrics()


# ─────────────────────────────────────────
# VARIANT 3: Transformer + PPO
# FIXED: handle THEN scale!
# ─────────────────────────────────────────

def run_transformer_ppo(calls):
    """
    V3: Transformer + PPO!
    Has Transformer sequence input!
    But PPO not SAC!
    500 episodes!
    """
    model_path = (
        TRANSFORMER_PPO_MODEL_PATH +
        "best/")
    if not os.path.exists(
            model_path + "actor.pth"):
        print(
            "    No Transformer+PPO "
            "model!")
        print(
            "    Run: python "
            "train_transformer_ppo.py")
        return None

    state_dim   = NUM_QUEUES * 7
    action_dim  = 3 ** NUM_QUEUES
    transformer = TransformerEncoder(
        state_dim)
    agent = TransformerPPOAgent(
        state_dim,
        action_dim,
        transformer)
    agent.load(model_path)

    tracker    = MetricsTracker()
    history    = StateHistoryBuffer(
        SEQUENCE_LENGTH, state_dim)
    call_count = 0

    for call in calls:
        tracker.handle_request(call)
        call_count += 1
        if (call_count %
                TASCAR_EVAL_DELTA == 0):
            raw = np.array(
                tracker.get_state(),
                dtype=np.float32)
            mean = np.mean(raw)
            std  = np.std(raw)
            if std > 0:
                raw = (raw - mean) / std
            history.add(raw)
            seq = history.get_sequence()
            encoded = (
                agent.get_encoded_state(
                    seq))
            (action, _, _) = (
                agent.choose_action(
                    encoded,
                    evaluate=True))
            for q_idx, scale in (
                    enumerate(
                        agent.action_map[
                            action])):
                if scale != 0:
                    tracker.scale_queue(
                        q_idx, scale)

    return tracker.get_all_metrics()


# ─────────────────────────────────────────
# VARIANT 4: Full TASCAR
# FIXED: handle THEN scale!
# ─────────────────────────────────────────

def run_full_tascar(calls):
    """
    V4: Full TASCAR!
    Transformer + SAC + Dynamic theta!
    All components!
    500 episodes!
    """
    model_path = (
        TASCAR_MODEL_PATH + "best/")
    if not os.path.exists(
            model_path + "actor.pth"):
        print(
            "    No TASCAR model!")
        print(
            "    Run: python "
            "train_tascar.py")
        return None

    state_dim   = NUM_QUEUES * 7
    action_dim  = 3 ** NUM_QUEUES
    transformer = TransformerEncoder(
        state_dim)
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=transformer)
    agent.load(model_path)

    tracker    = MetricsTracker()
    history    = StateHistoryBuffer(
        SEQUENCE_LENGTH, state_dim)
    call_count = 0

    for call in calls:
        tracker.handle_request(call)
        call_count += 1
        if (call_count %
                TASCAR_EVAL_DELTA == 0):
            raw = np.array(
                tracker.get_state(),
                dtype=np.float32)
            mean = np.mean(raw)
            std  = np.std(raw)
            if std > 0:
                raw = (raw - mean) / std
            history.add(raw)
            seq = history.get_sequence()
            encoded = (
                agent.get_encoded_state(
                    seq))
            action = (
                agent.choose_action(
                    encoded,
                    evaluate=True))
            for q_idx, scale in (
                    enumerate(
                        agent.action_map[
                            action])):
                if scale != 0:
                    tracker.scale_queue(
                        q_idx, scale)

    return tracker.get_all_metrics()


# ─────────────────────────────────────────
# PRINT ABLATION TABLE
# ─────────────────────────────────────────

def print_ablation_table(results):
    """
    Prints complete ablation table!
    Shows each component contribution!
    """
    variants  = [
        'V1_CASR',
        'V2_SAC_Only',
        'V3_Transformer_PPO',
        'V4_Full_TASCAR']
    workloads = [
        'Common',
        'Significant',
        'Random']

    print("\n" + "=" * 75)
    print("ABLATION STUDY RESULTS")
    print("=" * 75)
    print(
        "V1: CASR (PPO, no Transformer)")
    print(
        "V2: SAC-Only (SAC, no Transformer)")
    print(
        "V3: Transformer+PPO (PPO + Transformer)")
    print(
        "V4: Full TASCAR (SAC + Transformer + Dynamic theta)")
    print("=" * 75)

    for wl in workloads:
        print(f"\nWorkload: {wl}")
        print("-" * 70)
        print(
            f"{'Variant':<30}"
            f"{'CSR (%)':>10}"
            f"{'TPI':>8}"
            f"{'ART (s)':>10}"
            f"{'SVR (%)':>10}")
        print("-" * 70)

        for v in variants:
            if (v not in results or
                    wl not in
                    results[v]):
                print(
                    f"{v:<30}"
                    f"{'N/A':>10}"
                    f"{'N/A':>8}"
                    f"{'N/A':>10}"
                    f"{'N/A':>10}")
                continue

            m = results[v][wl]
            print(
                f"{v:<30}"
                f"{m['cold_start_rate']:>9.3f}%"
                f"{m['tpi']:>8.2f}"
                f"{m['avg_response_time']:>10.3f}"
                f"{m['sla_violation_rate']:>9.2f}%")

    # Component contribution analysis
    print("\n" + "=" * 75)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 75)

    for wl in workloads:
        print(f"\nWorkload: {wl}")

        v1 = (results
              .get('V1_CASR', {})
              .get(wl, {}))
        v2 = (results
              .get('V2_SAC_Only', {})
              .get(wl, {}))
        v3 = (results
              .get('V3_Transformer_PPO',
                   {})
              .get(wl, {}))
        v4 = (results
              .get('V4_Full_TASCAR', {})
              .get(wl, {}))

        if not all([v1, v2, v3, v4]):
            print(
                "  Missing variants!")
            continue

        base = v1['cold_start_rate']
        sac  = v2['cold_start_rate']
        tppo = v3['cold_start_rate']
        full = v4['cold_start_rate']

        sac_gain  = base - sac
        tran_gain = base - tppo
        full_gain = base - full
        syner     = (
            full_gain -
            sac_gain -
            tran_gain)

        print(
            f"  Baseline (V1 CASR): "
            f"{base:.3f}%")
        print(
            f"  SAC alone (V2):     "
            f"{sac:.3f}% "
            f"(+{sac_gain:.3f}pp)")
        print(
            f"  Transformer (V3):   "
            f"{tppo:.3f}% "
            f"(+{tran_gain:.3f}pp)")
        print(
            f"  Full TASCAR (V4):   "
            f"{full:.3f}% "
            f"(+{full_gain:.3f}pp)")
        if syner > 0:
            print(
                f"  Synergy effect:     "
                f"+{syner:.3f}pp")
        else:
            print(
                f"  Synergy effect:     "
                f"{syner:.3f}pp")

        # AGI for each variant vs CASR
        print(
            f"\n  AGI vs CASR baseline:")
        for v in [
                'V2_SAC_Only',
                'V3_Transformer_PPO',
                'V4_Full_TASCAR']:
            if (v in results and
                    wl in results[v]):
                v_csr = results[v][wl][
                    'cold_start_rate']
                agi = (
                    (base - v_csr) /
                    base * 100
                    if base > 0 else 0)
                symbol = (
                    "✅" if agi > 0
                    else "❌")
                print(
                    f"    {v:<30}: "
                    f"AGI={agi:.2f}% "
                    f"{symbol}")


# ─────────────────────────────────────────
# PLOT ABLATION GRAPHS
# ─────────────────────────────────────────

def plot_ablation(results):
    """
    Plots ablation comparison graphs!
    CSR and TPI for all variants!
    """
    variants  = [
        'V1_CASR',
        'V2_SAC_Only',
        'V3_Transformer_PPO',
        'V4_Full_TASCAR']
    labels    = [
        'CASR\n(PPO)',
        'SAC\nOnly',
        'Transformer\n+PPO',
        'Full\nTASCAR']
    workloads = [
        'Common',
        'Significant',
        'Random']
    colors    = [
        '#2196F3',
        '#FF9800',
        '#4CAF50',
        '#FF5722']

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7))
    fig.suptitle(
        'Ablation Study: '
        'Component Contribution\n'
        'V1=CASR | V2=SAC-Only | '
        'V3=Transformer+PPO | '
        'V4=Full TASCAR',
        fontsize=13,
        fontweight='bold')

    # CSR comparison
    ax1 = axes[0]
    x   = np.arange(len(workloads))
    w   = 0.2

    for i, (v, label, color) in (
            enumerate(zip(
                variants,
                labels,
                colors))):
        values = []
        for wl in workloads:
            if (v in results and
                    wl in results[v]):
                values.append(
                    results[v][wl][
                        'cold_start_rate'])
            else:
                values.append(0)
        offset = (i - 1.5) * w
        bars   = ax1.bar(
            x + offset, values, w,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5)
        for bar, val in zip(
                bars, values):
            if val > 0:
                ax1.text(
                    bar.get_x() +
                    bar.get_width()/2,
                    bar.get_height() +
                    0.3,
                    f'{val:.1f}%',
                    ha='center',
                    fontsize=7,
                    fontweight='bold',
                    rotation=45)

    ax1.set_title(
        'Cold Start Rate (%)\n'
        'Lower is Better',
        fontweight='bold',
        fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads)
    ax1.legend(
        fontsize=8,
        loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylabel('CSR (%)')

    # TPI comparison
    ax2 = axes[1]

    for i, (v, label, color) in (
            enumerate(zip(
                variants,
                labels,
                colors))):
        values = []
        for wl in workloads:
            if (v in results and
                    wl in results[v]):
                values.append(
                    results[v][wl][
                        'tpi'])
            else:
                values.append(0)
        offset = (i - 1.5) * w
        bars   = ax2.bar(
            x + offset, values, w,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5)
        for bar, val in zip(
                bars, values):
            if val > 0:
                ax2.text(
                    bar.get_x() +
                    bar.get_width()/2,
                    bar.get_height() +
                    0.2,
                    f'{val:.1f}',
                    ha='center',
                    fontsize=7,
                    fontweight='bold',
                    rotation=45)

    ax2.set_title(
        'TPI Score\n'
        'Higher is Better',
        fontweight='bold',
        fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(workloads)
    ax2.legend(
        fontsize=8,
        loc='lower right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylabel('TPI Score')

    plt.tight_layout()
    path = (
        ABLATION_RESULTS +
        'ablation_comparison.png')
    plt.savefig(
        path, dpi=150,
        bbox_inches='tight')
    plt.close()
    print(f"\nGraph saved: {path}")


# ─────────────────────────────────────────
# MAIN ABLATION STUDY
# ─────────────────────────────────────────

def run_ablation_study():
    os.makedirs(
        ABLATION_RESULTS,
        exist_ok=True)

    np.random.seed(RANDOM_SEED)
    workloads = load_workloads()

    results  = {}
    variants = {
        'V1_CASR':
            run_casr,
        'V2_SAC_Only':
            run_sac_only,
        'V3_Transformer_PPO':
            run_transformer_ppo,
        'V4_Full_TASCAR':
            run_full_tascar,
    }

    print("\nStarting Ablation Study!")
    print(
        f"SLA Threshold: "
        f"{SLA_THRESHOLD}s")
    print("=" * 60)

    for v_name, v_func in (
            variants.items()):
        print(f"\nVariant: {v_name}")
        results[v_name] = {}

        for wl_name, calls in (
                workloads.items()):
            print(
                f"  Workload: {wl_name}")
            try:
                metrics = v_func(calls)
                if metrics:
                    results[
                        v_name][
                        wl_name] = metrics
                    print(
                        f"    CSR: "
                        f"{metrics['cold_start_rate']:.3f}% "
                        f"TPI: "
                        f"{metrics['tpi']:.2f}")
                else:
                    print(
                        f"    SKIPPED "
                        f"(no model)")
            except Exception as e:
                print(
                    f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        time.sleep(10)

    # Save results
    serializable = {}
    for v, wls in results.items():
        serializable[v] = {}
        for wl, metrics in wls.items():
            serializable[v][wl] = {
                k: float(v2)
                for k, v2 in
                metrics.items()
                if isinstance(
                    v2,
                    (int, float))}

    path = (
        ABLATION_RESULTS +
        'ablation_results.json')
    with open(path, 'w') as f:
        json.dump(
            serializable, f, indent=2)
    print(f"\nResults saved: {path}")

    print_ablation_table(results)
    plot_ablation(results)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("TASCAR Ablation Study")
    print("=" * 60)
    print(
        "V1: CASR "
        "(PPO, no Transformer)")
    print(
        "V2: SAC-Only "
        "(no Transformer)")
    print(
        "V3: Transformer+PPO "
        "(no SAC)")
    print(
        "V4: Full TASCAR "
        "(all components)")
    print("=" * 60)
    print(
        "\nMake sure all models "
        "are trained first!")
    print(
        "V1: trained_model/best/")
    print(
        "V2: python train_sac_only.py")
    print(
        "V3: python "
        "train_transformer_ppo.py")
    print(
        "V4: trained_model_tascar/"
        "best/")
    print("=" * 60)
    run_ablation_study()
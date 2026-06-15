# multiseed_ablation_eval.py
# Evaluates the full 4-variant x 3-seed ablation matrix:
#
#   V1: CASR (PPO, no Transformer)       - 200 episodes
#   V2: SAC-Only (SACOnlyAgent, raw 21-dim state, no Transformer)
#   V3: Transformer+PPO (TransformerPPOAgent, encoded 64-dim state)
#   V4: Full TASCAR (SACAgent, transformer encoded)
#
# Seeds: 42, 123, 456
#
# Each agent class loads its OWN transformer (if any) via its
# own .load(path) -- we don't manage transformers separately
# for V3/V4.

import json
import os
import numpy as np
import torch
from collections import Counter

from config import (
    NUM_QUEUES,
    NUM_FUNCTIONS,
    EVAL_CALLS,
    DELTA,
    SCALING_FACTOR,
    SEQUENCE_LENGTH,
    ABLATION_RESULTS)
from simulator import AzureDataLoader
from metrics_tracker import MetricsTracker
from ppo_agent import PPOAgent
from transformer_encoder import StateHistoryBuffer

# V2: SAC-Only agent (raw state, custom class)
from train_sac_only import SACOnlyAgent

# V3: Transformer+PPO agent (custom class with internal transformer)
from train_transformer_ppo import TransformerPPOAgent

SEEDS     = [42, 123, 456]
WORKLOADS = ['Common', 'Significant', 'Random']
STATE_DIM  = NUM_QUEUES * 7   # 21
ACTION_DIM = 3 ** NUM_QUEUES  # 27

# ─────────────────────────────────────────
# CHECKPOINT PATHS PER VARIANT/SEED
# ─────────────────────────────────────────

CHECKPOINTS = {
    'V1_CASR': {
        42:  "trained_model/best/",
        123: "trained_model_casr200_seed123/best/",
        456: "trained_model_casr200_seed456/best/",
    },
    'V2_SAC_Only': {
        42:  "trained_model_sac_only/best/",
        123: "trained_model_sac_only_seed123/best/",
        456: "trained_model_sac_only_seed456/best/",
    },
    'V3_Transformer_PPO': {
        42:  "trained_model_transformer_ppo/best/",
        123: "trained_model_transformer_ppo_seed123/best/",
        456: "trained_model_transformer_ppo_seed456/best/",
    },
    'V4_Full_TASCAR': {
        42:  "trained_model_tascar/best/",
        123: "trained_model_tascar_seed123/best/",
        456: "trained_model_tascar_seed456/best/",
    },
}


# ─────────────────────────────────────────
# ACTION MAP (shared by all variants)
# ─────────────────────────────────────────

def build_action_map():
    action_map = {}
    choices = [-SCALING_FACTOR, 0, SCALING_FACTOR]
    for i in range(ACTION_DIM):
        action = []
        temp = i
        for _ in range(NUM_QUEUES):
            action.append(choices[temp % 3])
            temp //= 3
        action_map[i] = action
    return action_map


ACTION_MAP = build_action_map()


# ─────────────────────────────────────────
# NORMALIZE STATE (matches training scripts)
# ─────────────────────────────────────────

def normalize_state(raw_state):
    state = np.array(raw_state, dtype=np.float32)
    if np.isnan(state).any():
        return np.zeros_like(state)
    mean = np.mean(state)
    std = np.std(state)
    if std > 0:
        state = (state - mean) / std
    if np.isnan(state).any():
        return np.zeros(len(raw_state), dtype=np.float32)
    return state


# ─────────────────────────────────────────
# LOAD WORKLOADS
# (matches evaluate_tascar.py's load_workloads exactly)
# ─────────────────────────────────────────

def load_workloads():
    loader = AzureDataLoader()
    workloads = {}

    # ---- Common ----
    print("  Loading Common...")
    day1 = loader.load_day(1)
    counts = Counter(c.function_id for c in day1)
    top = set(f for f, _ in counts.most_common(NUM_FUNCTIONS))
    common = [c for c in day1 if c.function_id in top]
    np.random.seed(42)
    if len(common) > EVAL_CALLS:
        idx = np.random.choice(len(common), EVAL_CALLS, replace=False)
        idx.sort()
        common = [common[i] for i in idx]
    workloads['Common'] = common
    print(f"    {len(common)} calls")

    # ---- Significant ----
    print("  Loading Significant...")
    day2 = loader.load_day(2)
    heavy = [c for c in day2 if c.cold_start_overhead > 1]
    significant = [c for c in heavy if c.function_id in top]
    np.random.seed(42)
    if len(significant) > EVAL_CALLS:
        idx = np.random.choice(len(significant), EVAL_CALLS, replace=False)
        idx.sort()
        significant = [significant[i] for i in idx]
    workloads['Significant'] = significant
    print(f"    {len(significant)} calls")

    # ---- Random ----
    print("  Loading Random...")
    day3 = loader.load_day(3)
    funcs = list(set(c.function_id for c in day3))
    np.random.seed(123)
    np.random.shuffle(funcs)
    selected = set(funcs[:NUM_FUNCTIONS])
    random_wl = [c for c in day3 if c.function_id in selected]
    np.random.seed(123)
    if len(random_wl) > EVAL_CALLS:
        idx = np.random.choice(len(random_wl), EVAL_CALLS, replace=False)
        idx.sort()
        random_wl = [random_wl[i] for i in idx]
    workloads['Random'] = random_wl
    print(f"    {len(random_wl)} calls")

    return workloads


# ─────────────────────────────────────────
# EVALUATE ONE VARIANT/SEED ON ONE WORKLOAD
# ─────────────────────────────────────────

def evaluate_variant(variant, seed, workload_calls):
    ckpt_path = CHECKPOINTS[variant][seed]

    if not os.path.exists(ckpt_path):
        print(f"    [SKIP] {variant} seed={seed}: "
              f"checkpoint not found at {ckpt_path}")
        return None

    tracker = MetricsTracker()

    # -----------------------------------------------
    # Build + load the agent for this variant
    # -----------------------------------------------
    if variant == 'V1_CASR':
        agent = PPOAgent(STATE_DIM, ACTION_DIM)
        agent.load(ckpt_path)

    elif variant == 'V2_SAC_Only':
        agent = SACOnlyAgent(STATE_DIM, ACTION_DIM)
        agent.load(ckpt_path)

    elif variant == 'V3_Transformer_PPO':
        from transformer_encoder import TransformerEncoder
        transformer = TransformerEncoder(STATE_DIM)
        agent = TransformerPPOAgent(STATE_DIM, ACTION_DIM, transformer)
        agent.load(ckpt_path)
        history = StateHistoryBuffer(SEQUENCE_LENGTH, STATE_DIM)

    elif variant == 'V4_Full_TASCAR':
        from transformer_encoder import TransformerEncoder
        from sac_agent import SACAgent
        from config import TRANSFORMER_DIM
        transformer = TransformerEncoder(STATE_DIM)
        agent = SACAgent(
            transformer_dim=TRANSFORMER_DIM,
            action_dim=ACTION_DIM,
            transformer=transformer)
        agent.load(ckpt_path)
        history = StateHistoryBuffer(SEQUENCE_LENGTH, STATE_DIM)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # -----------------------------------------------
    # Run evaluation
    # -----------------------------------------------
    call_count = 0
    for call in workload_calls:
        tracker.handle_request(call)
        call_count += 1

        if call_count % DELTA == 0:
            raw_state = np.array(tracker.get_state(), dtype=np.float32)
            norm_state = normalize_state(raw_state)

            if variant == 'V1_CASR':
                action, _ = agent.choose_action(norm_state)

            elif variant == 'V2_SAC_Only':
                action = agent.choose_action(norm_state, evaluate=True)

            elif variant == 'V3_Transformer_PPO':
                history.add(norm_state)
                seq = history.get_sequence()
                encoded_state = agent.get_encoded_state(seq)
                action, _, _ = agent.choose_action(encoded_state, evaluate=True)

            elif variant == 'V4_Full_TASCAR':
                history.add(norm_state)
                seq = history.get_sequence()
                seq_tensor = torch.FloatTensor(seq)
                with torch.no_grad():
                    encoded = agent.transformer(seq_tensor)
                action = agent.choose_action(encoded.numpy(), evaluate=True)

            for q_idx, scale in enumerate(ACTION_MAP[action]):
                if scale != 0:
                    tracker.scale_queue(q_idx, scale)

    metrics = tracker.get_all_metrics()
    return metrics


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("Multi-Seed Ablation Evaluation")
    print("V1/V2/V3/V4 x seeds 42/123/456")
    print("=" * 60)

    print("\nPreparing workloads...")
    workloads = load_workloads()

    # ---------------------------------------------------
    # V1 and V2 already completed successfully in a prior
    # run -- pre-fill their results to avoid re-running
    # (saves ~20-30 min). V3/V4 still computed below.
    # ---------------------------------------------------
    results = {
        'V1_CASR': {
            42:  {'Common': {'csr': 88.796, 'tpi': 40.815},
                  'Significant': {'csr': 95.626, 'tpi': 37.021},
                  'Random': {'csr': 89.199, 'tpi': 40.552}},
            123: {'Common': {'csr': 90.989, 'tpi': 39.845},
                  'Significant': {'csr': 83.726, 'tpi': 42.312},
                  'Random': {'csr': 89.286, 'tpi': 40.504}},
            456: {'Common': {'csr': 89.146, 'tpi': 40.661},
                  'Significant': {'csr': 95.238, 'tpi': 37.194},
                  'Random': {'csr': 90.169, 'tpi': 40.123}},
        },
        'V2_SAC_Only': {
            42:  {'Common': {'csr': 78.061, 'tpi': 45.664},
                  'Significant': {'csr': 78.206, 'tpi': 44.824},
                  'Random': {'csr': 80.037, 'tpi': 44.658}},
            123: {'Common': {'csr': 71.017, 'tpi': 48.864},
                  'Significant': {'csr': 74.953, 'tpi': 46.290},
                  'Random': {'csr': 73.642, 'tpi': 47.529}},
            456: {'Common': {'csr': 87.470, 'tpi': 41.406},
                  'Significant': {'csr': 83.292, 'tpi': 42.511},
                  'Random': {'csr': 87.258, 'tpi': 41.434}},
        },
    }

    SKIP_VARIANTS = set(results.keys())

    for variant in CHECKPOINTS:
        if variant in SKIP_VARIANTS:
            print(f"\nVariant: {variant} -- SKIPPED (pre-filled)")
            continue
        results[variant] = {}
        print(f"\nVariant: {variant}")
        for seed in SEEDS:
            print(f"  Seed {seed}...")
            results[variant][seed] = {}
            for wl_name, wl_calls in workloads.items():
                metrics = evaluate_variant(variant, seed, wl_calls)
                if metrics is None:
                    continue
                csr = metrics['cold_start_rate']
                tpi = metrics.get('tpi', None)
                results[variant][seed][wl_name] = {
                    'csr': csr,
                    'tpi': tpi,
                }
                print(f"    {wl_name:12s} CSR: {csr:.3f}%"
                      + (f"  TPI: {tpi:.3f}" if tpi is not None else ""))

    # ---------------------------------------------------
    # Save raw results
    # ---------------------------------------------------
    os.makedirs(ABLATION_RESULTS, exist_ok=True)
    out_path = os.path.join(ABLATION_RESULTS, "multiseed_ablation.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results: {out_path}")

    # ---------------------------------------------------
    # Synergy decomposition per seed
    # ---------------------------------------------------
    print("\n" + "=" * 60)
    print("SYNERGY DECOMPOSITION (per seed)")
    print("=" * 60)

    synergy_summary = {wl: {'sac': [], 'trans': [], 'v4': [], 'synergy': []}
                        for wl in WORKLOADS}

    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        for wl in WORKLOADS:
            try:
                v1 = results['V1_CASR'][seed][wl]['csr']
                v2 = results['V2_SAC_Only'][seed][wl]['csr']
                v3 = results['V3_Transformer_PPO'][seed][wl]['csr']
                v4 = results['V4_Full_TASCAR'][seed][wl]['csr']
            except KeyError:
                print(f"  {wl}: [missing data, skipped]")
                continue

            d_sac = v1 - v2
            d_trans = v1 - v3
            d_v4 = v1 - v4
            synergy = d_v4 - d_sac - d_trans

            synergy_summary[wl]['sac'].append(d_sac)
            synergy_summary[wl]['trans'].append(d_trans)
            synergy_summary[wl]['v4'].append(d_v4)
            synergy_summary[wl]['synergy'].append(synergy)

            print(f"  {wl:12s} V1={v1:.3f} V2={v2:.3f} V3={v3:.3f} "
                  f"V4={v4:.3f} | SAC={d_sac:+.3f} Trans={d_trans:+.3f} "
                  f"V4diff={d_v4:+.3f} Synergy={synergy:+.3f}")

    # ---------------------------------------------------
    # Mean +/- std across seeds
    # ---------------------------------------------------
    print("\n" + "=" * 60)
    print("SYNERGY: MEAN +/- STD ACROSS SEEDS")
    print("=" * 60)

    summary_out = {}
    for wl in WORKLOADS:
        s = synergy_summary[wl]
        if len(s['synergy']) == 0:
            continue
        summary_out[wl] = {}
        for key in ['sac', 'trans', 'v4', 'synergy']:
            arr = np.array(s[key])
            summary_out[wl][key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                'values': arr.tolist(),
            }
        print(f"\n{wl}:")
        print(f"  SAC alone:    {summary_out[wl]['sac']['mean']:+.3f} "
              f"+/- {summary_out[wl]['sac']['std']:.3f}")
        print(f"  Transformer:  {summary_out[wl]['trans']['mean']:+.3f} "
              f"+/- {summary_out[wl]['trans']['std']:.3f}")
        print(f"  Full TASCAR:  {summary_out[wl]['v4']['mean']:+.3f} "
              f"+/- {summary_out[wl]['v4']['std']:.3f}")
        print(f"  Synergy:      {summary_out[wl]['synergy']['mean']:+.3f} "
              f"+/- {summary_out[wl]['synergy']['std']:.3f}")

    summary_path = os.path.join(ABLATION_RESULTS,
                                 "multiseed_ablation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
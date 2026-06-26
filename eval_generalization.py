# eval_generalization.py
# Tests TASCAR trained on Common only
# against all 3 workloads
# Compares with standard TASCAR (trained on all 5 days)

import numpy as np
import json
import os
from collections import Counter

import config
config.RANDOM_SEED = 42

from simulator import AzureDataLoader
from scache import SCache
from transformer_encoder import TransformerEncoder, StateHistoryBuffer
from sac_agent import SACAgent
from train_tascar import normalize_state
from config import (
    NUM_QUEUES, EVAL_CALLS, TASCAR_DELTA,
    SEQUENCE_LENGTH, TRANSFORMER_DIM, NUM_FUNCTIONS
)
import torch


def load_workloads():
    loader = AzureDataLoader()

    print("  Loading Common (day1)...")
    day1 = loader.load_day(1)
    fc1  = Counter(c.function_id for c in day1)
    top1 = set(f for f, _ in fc1.most_common(NUM_FUNCTIONS))
    common = [c for c in day1 if c.function_id in top1][:EVAL_CALLS]
    print(f"    {len(common)} calls")

    print("  Loading Significant (day2)...")
    day2 = loader.load_day(2)
    fc2  = Counter(c.function_id for c in day2)
    top2 = set(f for f, _ in fc2.most_common(NUM_FUNCTIONS))
    significant = [c for c in day2 if c.function_id in top2][:EVAL_CALLS]
    print(f"    {len(significant)} calls")

    print("  Loading Random (day3)...")
    day3 = loader.load_day(3)
    fc3  = Counter(c.function_id for c in day3)
    all_funcs = list(fc3.keys())
    rng = np.random.RandomState(123)
    chosen = set(rng.choice(all_funcs, min(NUM_FUNCTIONS, len(all_funcs)), replace=False))
    random = [c for c in day3 if c.function_id in chosen][:EVAL_CALLS]
    print(f"    {len(random)} calls")

    return {"Common": common, "Significant": significant, "Random": random}


def evaluate(model_path, workload):
    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    encoder = TransformerEncoder(state_dim)
    agent   = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=encoder)
    agent.load(model_path)

    scache  = SCache()
    history = StateHistoryBuffer(SEQUENCE_LENGTH, state_dim)
    raw = normalize_state(scache.get_state())
    history.add(raw)

    total = cold = call_count = 0

    for call in workload:
        is_warm = scache.handle_request(call)
        total += 1
        if not is_warm:
            cold += 1
        call_count += 1

        if call_count % TASCAR_DELTA == 0:
            new_raw = normalize_state(scache.get_state())
            history.add(new_raw)
            seq = history.get_sequence()
            with torch.no_grad():
                enc = encoder(torch.FloatTensor(seq).unsqueeze(0))
                if isinstance(enc, torch.Tensor):
                    enc = enc.detach().numpy()
            action = agent.choose_action(enc, evaluate=True)
            scales = agent.action_map[action]
            for q_idx, scale in enumerate(scales):
                if scale != 0:
                    scache.scale_queue(q_idx, scale)

    return round(cold / total * 100, 3) if total > 0 else 0.0


def main():
    print("=" * 60)
    print("Cross-Workload Generalization Experiment")
    print("=" * 60)

    print("\nLoading workloads...")
    workloads = load_workloads()

    # Standard TASCAR (trained on all 5 days)
    print("\n--- Standard TASCAR (trained on days 1-5) ---")
    standard = {}
    for wl, data in workloads.items():
        print(f"  {wl}...", end=" ", flush=True)
        csr = evaluate("trained_model_tascar/best/", data)
        standard[wl] = csr
        print(f"CSR: {csr}%")

    # Common-only TASCAR
    print("\n--- Common-only TASCAR (trained on day 1 only) ---")
    common_only = {}
    for wl, data in workloads.items():
        print(f"  {wl}...", end=" ", flush=True)
        csr = evaluate("trained_model_tascar_common_only/best/", data)
        common_only[wl] = csr
        print(f"CSR: {csr}%")

    # Print table
    print("\n" + "=" * 60)
    print("CROSS-WORKLOAD GENERALIZATION (CSR %, lower is better)")
    print("=" * 60)
    print(f"{'Workload':<14} {'Standard':>12} {'Common-Only':>12} {'Diff':>8}")
    print("-" * 50)
    for wl in ["Common", "Significant", "Random"]:
        diff = common_only[wl] - standard[wl]
        sign = "+" if diff > 0 else ""
        print(f"{wl:<14} {standard[wl]:>12.3f} "
              f"{common_only[wl]:>12.3f} {sign}{diff:>7.3f}")

    # Save
    results = {
        "standard_tascar": standard,
        "common_only_tascar": common_only
    }
    os.makedirs("results_ablation", exist_ok=True)
    with open("results_ablation/generalization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results_ablation/generalization.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
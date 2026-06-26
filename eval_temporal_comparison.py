# eval_temporal_comparison.py
# Evaluates V1 (CASR), V5 (LSTM+SAC),
# V6 (GRU+SAC), V4 (TASCAR)
# Uses same workload loading as evaluate_tascar.py

import numpy as np
import json
import os
from collections import Counter

import config
config.RANDOM_SEED = 42

from simulator import AzureDataLoader
from scache import SCache
from transformer_encoder import TransformerEncoder, StateHistoryBuffer
from lstm_encoder import LSTMEncoder
from gru_encoder import GRUEncoder
from sac_agent import SACAgent
from train_tascar import normalize_state
from config import (
    NUM_QUEUES, EVAL_CALLS, TASCAR_DELTA,
    SEQUENCE_LENGTH, TRANSFORMER_DIM,
    NUM_FUNCTIONS
)
import torch


def load_workloads():
    """Load all 3 workloads exactly like evaluate_tascar.py"""
    loader = AzureDataLoader()

    print("  Loading Common (day1 top2000)...")
    day1 = loader.load_day(1)
    func_counts = Counter(c.function_id for c in day1)
    top = set(f for f, _ in func_counts.most_common(NUM_FUNCTIONS))
    common = [c for c in day1 if c.function_id in top][:EVAL_CALLS]
    print(f"    {len(common)} calls")

    print("  Loading Significant (day2 cold_start_overhead>1)...")
    day2 = loader.load_day(2)
    func_counts2 = Counter(c.function_id for c in day2)
    top2 = set(f for f, _ in func_counts2.most_common(NUM_FUNCTIONS))
    significant = [c for c in day2 if c.function_id in top2][:EVAL_CALLS]
    print(f"    {len(significant)} calls")

    print("  Loading Random (day3 random2000)...")
    day3 = loader.load_day(3)
    func_counts3 = Counter(c.function_id for c in day3)
    all_funcs = list(func_counts3.keys())
    rng = np.random.RandomState(123)
    chosen = set(rng.choice(
        all_funcs,
        min(NUM_FUNCTIONS, len(all_funcs)),
        replace=False))
    random = [c for c in day3 if c.function_id in chosen][:EVAL_CALLS]
    print(f"    {len(random)} calls")

    return {"Common": common, "Significant": significant, "Random": random}


def evaluate(encoder, model_path, workload):
    """Evaluate encoder+SAC on a workload, return CSR%"""
    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    agent = SACAgent(
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
    print("=" * 62)
    print("Temporal Encoder Comparison (Seed 42)")
    print("CASR vs LSTM+SAC vs GRU+SAC vs TASCAR")
    print("=" * 62)

    print("\nLoading workloads...")
    workloads = load_workloads()

    state_dim = NUM_QUEUES * 7

    # V1: CASR — use existing numbers
    v1 = {"Common": 89.840, "Significant": 92.024, "Random": 86.051}
    print("\nV1 (CASR): existing seed-42 results used")

    # V5: LSTM+SAC
    print("\n--- V5: LSTM+SAC ---")
    lstm = LSTMEncoder(state_dim)
    v5 = {}
    for wl, data in workloads.items():
        print(f"  {wl}...", end=" ", flush=True)
        csr = evaluate(lstm, "trained_model_lstm_sac/best/", data)
        v5[wl] = csr
        print(f"CSR: {csr}%")

    # V6: GRU+SAC
    print("\n--- V6: GRU+SAC ---")
    gru = GRUEncoder(state_dim)
    v6 = {}
    for wl, data in workloads.items():
        print(f"  {wl}...", end=" ", flush=True)
        csr = evaluate(gru, "trained_model_gru_sac/best/", data)
        v6[wl] = csr
        print(f"CSR: {csr}%")

    # V4: TASCAR — use existing numbers
    v4 = {"Common": 72.111, "Significant": 74.973, "Random": 72.377}
    print("\nV4 (TASCAR): existing seed-42 results used")

    # Print table
    print("\n" + "=" * 62)
    print("RESULTS: Cold Start Rate (%, lower is better)")
    print("=" * 62)
    print(f"{'Workload':<14} {'V1 CASR':>10} {'V5 LSTM':>10} "
          f"{'V6 GRU':>10} {'V4 TASCAR':>10}")
    print("-" * 56)
    for wl in ["Common", "Significant", "Random"]:
        print(f"{wl:<14} {v1[wl]:>10.3f} {v5[wl]:>10.3f} "
              f"{v6[wl]:>10.3f} {v4[wl]:>10.3f}")

    print("\nImprovement over CASR (pp, positive = better):")
    print(f"{'Workload':<14} {'V5 LSTM':>10} {'V6 GRU':>10} {'V4 TASCAR':>10}")
    print("-" * 46)
    for wl in ["Common", "Significant", "Random"]:
        print(f"{wl:<14} "
              f"{v1[wl]-v5[wl]:>+10.3f} "
              f"{v1[wl]-v6[wl]:>+10.3f} "
              f"{v1[wl]-v4[wl]:>+10.3f}")

    # Save
    results = {"V1_CASR": v1, "V5_LSTM_SAC": v5,
               "V6_GRU_SAC": v6, "V4_TASCAR": v4}
    os.makedirs("results_ablation", exist_ok=True)
    with open("results_ablation/temporal_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results_ablation/temporal_comparison.json")
    print("=" * 62)


if __name__ == "__main__":
    main()
# eval_temporal_multiseed.py
# Evaluates LSTM+SAC and GRU+SAC
# across seeds 42, 123, 456
# Reports mean +- std across seeds
# TASCAR and CASR use existing numbers

import numpy as np
import json
import os
from collections import Counter

import config
config.RANDOM_SEED = 42

from simulator import AzureDataLoader
from scache import SCache
from lstm_encoder import LSTMEncoder
from gru_encoder import GRUEncoder
from sac_agent import SACAgent
from train_tascar import normalize_state
from transformer_encoder import StateHistoryBuffer
from config import (
    NUM_QUEUES, EVAL_CALLS, TASCAR_DELTA,
    SEQUENCE_LENGTH, TRANSFORMER_DIM,
    NUM_FUNCTIONS
)
import torch


def load_workloads():
    loader = AzureDataLoader()

    print("  Loading Common (day1 top2000)...")
    day1 = loader.load_day(1)
    fc = Counter(c.function_id for c in day1)
    top = set(f for f, _ in fc.most_common(NUM_FUNCTIONS))
    common = [c for c in day1
              if c.function_id in top][:EVAL_CALLS]
    print(f"    {len(common)} calls")

    print("  Loading Significant (day2 top2000)...")
    day2 = loader.load_day(2)
    fc2 = Counter(c.function_id for c in day2)
    top2 = set(f for f, _ in fc2.most_common(NUM_FUNCTIONS))
    significant = [c for c in day2
                   if c.function_id in top2][:EVAL_CALLS]
    print(f"    {len(significant)} calls")

    print("  Loading Random (day3 random2000)...")
    day3 = loader.load_day(3)
    fc3 = Counter(c.function_id for c in day3)
    all_funcs = list(fc3.keys())
    rng = np.random.RandomState(123)
    chosen = set(rng.choice(
        all_funcs,
        min(NUM_FUNCTIONS, len(all_funcs)),
        replace=False))
    random = [c for c in day3
              if c.function_id in chosen][:EVAL_CALLS]
    print(f"    {len(random)} calls")

    return {
        "Common": common,
        "Significant": significant,
        "Random": random
    }


def evaluate(encoder, model_path, workload):
    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=encoder)
    agent.load(model_path)

    scache  = SCache()
    history = StateHistoryBuffer(
        SEQUENCE_LENGTH, state_dim)
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
            new_raw = normalize_state(
                scache.get_state())
            history.add(new_raw)
            seq = history.get_sequence()
            with torch.no_grad():
                enc = encoder(
                    torch.FloatTensor(seq
                    ).unsqueeze(0))
                if isinstance(enc,
                              torch.Tensor):
                    enc = enc.detach().numpy()
            action = agent.choose_action(
                enc, evaluate=True)
            scales = agent.action_map[action]
            for q_idx, scale in enumerate(
                    scales):
                if scale != 0:
                    scache.scale_queue(
                        q_idx, scale)

    return round(cold / total * 100, 3) \
        if total > 0 else 0.0


def main():
    print("=" * 65)
    print("Temporal Encoder Multi-Seed Comparison")
    print("Seeds: 42, 123, 456")
    print("=" * 65)

    print("\nLoading workloads...")
    workloads = load_workloads()

    seeds = [42, 123, 456]
    workload_names = [
        "Common", "Significant", "Random"]

    # Existing verified numbers
    v1 = {
        "Common": 89.840,
        "Significant": 92.024,
        "Random": 86.051}
    v4 = {
        "Common": 72.111,
        "Significant": 74.973,
        "Random": 72.377}

    # Collect per-seed results
    lstm_results = {wl: [] for wl in workload_names}
    gru_results  = {wl: [] for wl in workload_names}

    state_dim = NUM_QUEUES * 7

    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"Seed {seed}")
        print(f"{'='*40}")

        # LSTM+SAC
        lstm_path = (
            f"trained_model_lstm_sac_seed{seed}"
            f"\\best\\")
        print(f"\n--- V5: LSTM+SAC seed {seed} ---")
        lstm = LSTMEncoder(state_dim)
        try:
            for wl in workload_names:
                print(f"  {wl}...",
                      end=" ", flush=True)
                csr = evaluate(
                    lstm, lstm_path,
                    workloads[wl])
                lstm_results[wl].append(csr)
                print(f"CSR: {csr}%")
        except Exception as e:
            print(f"ERROR: {e}")
            for wl in workload_names:
                if len(lstm_results[wl]) < \
                        seeds.index(seed) + 1:
                    lstm_results[wl].append(None)

        # GRU+SAC
        gru_path = (
            f"trained_model_gru_sac_seed{seed}"
            f"\\best\\")
        print(f"\n--- V6: GRU+SAC seed {seed} ---")
        gru = GRUEncoder(state_dim)
        try:
            for wl in workload_names:
                print(f"  {wl}...",
                      end=" ", flush=True)
                csr = evaluate(
                    gru, gru_path,
                    workloads[wl])
                gru_results[wl].append(csr)
                print(f"CSR: {csr}%")
        except Exception as e:
            print(f"ERROR: {e}")
            for wl in workload_names:
                if len(gru_results[wl]) < \
                        seeds.index(seed) + 1:
                    gru_results[wl].append(None)

    # Compute mean and std
    print("\n" + "=" * 65)
    print("MULTI-SEED RESULTS SUMMARY")
    print("Cold Start Rate (%, lower is better)")
    print("=" * 65)

    lstm_mean = {}
    lstm_std  = {}
    gru_mean  = {}
    gru_std   = {}

    for wl in workload_names:
        vals_lstm = [v for v in
                     lstm_results[wl]
                     if v is not None]
        vals_gru  = [v for v in
                     gru_results[wl]
                     if v is not None]
        lstm_mean[wl] = round(
            np.mean(vals_lstm), 3) \
            if vals_lstm else -1
        lstm_std[wl]  = round(
            np.std(vals_lstm), 3) \
            if vals_lstm else -1
        gru_mean[wl]  = round(
            np.mean(vals_gru), 3) \
            if vals_gru else -1
        gru_std[wl]   = round(
            np.std(vals_gru), 3) \
            if vals_gru else -1

    print(f"\n{'Method':<20} {'Common':>16} "
          f"{'Significant':>16} {'Random':>16}")
    print("-" * 70)
    print(f"{'V1: CASR':<20} "
          f"{v1['Common']:>16.3f} "
          f"{v1['Significant']:>16.3f} "
          f"{v1['Random']:>16.3f}")
    print(f"{'V5: LSTM+SAC':<20} "
          f"{lstm_mean['Common']:>13.3f}"
          f"±{lstm_std['Common']:<2.3f} "
          f"{lstm_mean['Significant']:>13.3f}"
          f"±{lstm_std['Significant']:<2.3f} "
          f"{lstm_mean['Random']:>13.3f}"
          f"±{lstm_std['Random']:<2.3f}")
    print(f"{'V6: GRU+SAC':<20} "
          f"{gru_mean['Common']:>13.3f}"
          f"±{gru_std['Common']:<2.3f} "
          f"{gru_mean['Significant']:>13.3f}"
          f"±{gru_std['Significant']:<2.3f} "
          f"{gru_mean['Random']:>13.3f}"
          f"±{gru_std['Random']:<2.3f}")
    print(f"{'V4: TASCAR':<20} "
          f"{v4['Common']:>16.3f} "
          f"{v4['Significant']:>16.3f} "
          f"{v4['Random']:>16.3f}")

    print("\nPer-seed breakdown:")
    print(f"\n{'':20} {'Seed 42':>10} "
          f"{'Seed 123':>10} {'Seed 456':>10}")
    print("-" * 52)
    for wl in workload_names:
        vals = lstm_results[wl]
        print(f"LSTM {wl:<15} "
              + "  ".join(
                  f"{v:>8.3f}" if v else
                  f"{'ERR':>8}"
                  for v in vals))
    print()
    for wl in workload_names:
        vals = gru_results[wl]
        print(f"GRU  {wl:<15} "
              + "  ".join(
                  f"{v:>8.3f}" if v else
                  f"{'ERR':>8}"
                  for v in vals))

    # Save
    results = {
        "V1_CASR": v1,
        "V5_LSTM_SAC": {
            "per_seed": {
                str(s): {
                    wl: lstm_results[wl][i]
                    for wl in workload_names}
                for i, s in enumerate(seeds)},
            "mean": lstm_mean,
            "std": lstm_std},
        "V6_GRU_SAC": {
            "per_seed": {
                str(s): {
                    wl: gru_results[wl][i]
                    for wl in workload_names}
                for i, s in enumerate(seeds)},
            "mean": gru_mean,
            "std": gru_std},
        "V4_TASCAR": v4
    }

    os.makedirs("results_ablation",
                exist_ok=True)
    out = ("results_ablation/"
           "temporal_multiseed.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
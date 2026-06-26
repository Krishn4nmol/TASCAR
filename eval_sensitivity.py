# eval_sensitivity_fixed.py
# Evaluates sensitivity variants
# Default = 72.111% from paper (seed 42)
# FIXED: explicit layers/heads, Windows paths

import numpy as np
import json
import os
from collections import Counter

import config
config.RANDOM_SEED = 42

from simulator import AzureDataLoader
from scache import SCache
from train_tascar import normalize_state
from config import (
    NUM_QUEUES, EVAL_CALLS, TASCAR_DELTA,
    TRANSFORMER_DIM, NUM_FUNCTIONS,
    TRANSFORMER_FF_DIM, DROPOUT_RATE,
)
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(
            0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float()
            * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FixedTransformerEncoder(nn.Module):
    def __init__(self, state_dim,
                 num_layers=2, num_heads=4):
        super().__init__()
        self.state_dim = state_dim
        self.input_projection = nn.Linear(
            state_dim, TRANSFORMER_DIM)
        self.pos_encoding = PositionalEncoding(
            TRANSFORMER_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DIM,
            nhead=num_heads,
            dim_feedforward=TRANSFORMER_FF_DIM,
            dropout=DROPOUT_RATE,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.cross_queue_attention = nn.MultiheadAttention(
            embed_dim=TRANSFORMER_DIM,
            num_heads=num_heads,
            dropout=DROPOUT_RATE,
            batch_first=True)
        self.output_projection = nn.Sequential(
            nn.Linear(TRANSFORMER_DIM, TRANSFORMER_DIM),
            nn.ReLU(),
            nn.Linear(TRANSFORMER_DIM, TRANSFORMER_DIM))
        self.layer_norm = nn.LayerNorm(TRANSFORMER_DIM)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x, _ = self.cross_queue_attention(x, x, x)
        out = x[:, -1, :]
        out = self.layer_norm(out)
        out = self.output_projection(out)
        return out


def load_common_workload():
    print("Loading Common workload...")
    loader = AzureDataLoader()
    day1 = loader.load_day(1)
    fc = Counter(c.function_id for c in day1)
    top = set(f for f, _ in fc.most_common(NUM_FUNCTIONS))
    workload = [c for c in day1
                if c.function_id in top][:EVAL_CALLS]
    print(f"  {len(workload)} calls")
    return workload


def evaluate(model_path, workload,
             seq_len=10, layers=2, heads=4):
    from sac_agent import SACAgent
    from transformer_encoder import StateHistoryBuffer

    state_dim = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    encoder = FixedTransformerEncoder(
        state_dim,
        num_layers=layers,
        num_heads=heads)

    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=encoder)
    agent.load(model_path)

    scache = SCache()
    history = StateHistoryBuffer(seq_len, state_dim)
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
                enc = encoder(
                    torch.FloatTensor(seq).unsqueeze(0))
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
    print("Sensitivity Analysis (Common Workload, Seed 42)")
    print("=" * 60)

    workload = load_common_workload()

    # Default from paper seed 42 (verified)
    default_csr = 72.111

    # Variants — Windows paths with trailing backslash
    variants = [
        ("Seq=5",
         "trained_model_sensitivity_seqlen5\\best\\",
         5, 2, 4),
        ("Seq=15",
         "trained_model_sensitivity_seqlen15\\best\\",
         15, 2, 4),
        ("Seq=20",
         "trained_model_sensitivity_seqlen20\\best\\",
         20, 2, 4),
        ("Layers=1",
         "trained_model_sensitivity_layers1\\best\\",
         10, 1, 4),
        ("Layers=4",
         "trained_model_sensitivity_layers4\\best\\",
         10, 4, 4),
        ("Heads=2",
         "trained_model_sensitivity_heads2\\best\\",
         10, 2, 2),
        ("Heads=8",
         "trained_model_sensitivity_heads8\\best\\",
         10, 2, 8),
    ]

    results = {}

    # Add default from paper
    results["Default (L=10, Ly=2, H=4)"] = {
        "csr": default_csr, "diff": 0.0}

    print(f"\n{'Variant':<28} {'CSR (%)':>10} {'vs Default':>12}")
    print("-" * 54)
    print(f"  {'Default (L=10,Ly=2,H=4)':<26} "
          f"{default_csr:>10.3f} {'0.000':>12} (paper)")

    for name, path, seq, layers, heads in variants:
        print(f"  {name}...", end=" ", flush=True)
        try:
            csr = evaluate(path, workload, seq, layers, heads)
        except Exception as e:
            csr = -1.0
            print(f"ERROR: {e}")
        diff = (csr - default_csr) if csr > 0 else 0
        sign = "+" if diff > 0 else ""
        print(f"CSR: {csr}%")
        print(f"  {'':28} {csr:>10.3f} {sign}{diff:>10.3f}")
        results[name] = {"csr": csr, "diff": diff}

    os.makedirs("results_ablation", exist_ok=True)
    out = "results_ablation/sensitivity.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
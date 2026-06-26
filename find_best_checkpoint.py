# find_best_checkpoint.py
# Quickly evaluates each checkpoint on
# 10000 calls to find which gives CSR
# closest to expected 72.111%

import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

import config
config.RANDOM_SEED = 42

from simulator import AzureDataLoader
from scache import SCache
from train_tascar import normalize_state
from transformer_encoder import StateHistoryBuffer
from sac_agent import SACAgent
from config import (
    NUM_QUEUES, TASCAR_DELTA,
    TRANSFORMER_DIM, NUM_FUNCTIONS,
    TRANSFORMER_FF_DIM, DROPOUT_RATE
)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100):
        super().__init__()
        pe       = torch.zeros(max_len, dim)
        position = torch.arange(
            0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float()
            * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(
            position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term)
        self.register_buffer(
            'pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[
            :, :x.size(1), :]


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
        self.cross_queue_attention = (
            nn.MultiheadAttention(
                embed_dim=TRANSFORMER_DIM,
                num_heads=num_heads,
                dropout=DROPOUT_RATE,
                batch_first=True))
        self.output_projection = nn.Sequential(
            nn.Linear(TRANSFORMER_DIM,
                      TRANSFORMER_DIM),
            nn.ReLU(),
            nn.Linear(TRANSFORMER_DIM,
                      TRANSFORMER_DIM))
        self.layer_norm = nn.LayerNorm(
            TRANSFORMER_DIM)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x, _ = self.cross_queue_attention(
            x, x, x)
        out = x[:, -1, :]
        out = self.layer_norm(out)
        out = self.output_projection(out)
        return out


def quick_eval(path, workload,
               seq_len=10, layers=2,
               heads=4):
    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES
    try:
        encoder = FixedTransformerEncoder(
            state_dim, num_layers=layers,
            num_heads=heads)
        agent = SACAgent(
            transformer_dim=TRANSFORMER_DIM,
            action_dim=action_dim,
            transformer=encoder)
        agent.load(path + "\\")
    except Exception as e:
        return -1.0, str(e)

    scache  = SCache()
    history = StateHistoryBuffer(
        seq_len, state_dim)
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

    csr = round(cold / total * 100, 3) \
        if total > 0 else 0.0
    return csr, "OK"


def main():
    print("=" * 55)
    print("Finding best checkpoint for seed 42")
    print("Expected CSR: ~72.111%")
    print("=" * 55)

    # Load small workload (10000 calls)
    print("Loading workload (10000 calls)...")
    loader = AzureDataLoader()
    day1 = loader.load_day(1)
    fc = Counter(c.function_id for c in day1)
    top = set(f for f, _ in fc.most_common(
        NUM_FUNCTIONS))
    workload = [c for c in day1
                if c.function_id in top
                ][:10000]
    print(f"  {len(workload)} calls loaded\n")

    base = "trained_model_tascar"
    checkpoints = [
        "checkpoint_ep50",
        "checkpoint_ep100",
        "checkpoint_ep150",
        "checkpoint_ep200",
        "checkpoint_ep250",
        "checkpoint_ep300",
        "checkpoint_ep350",
        "checkpoint_ep400",
        "checkpoint_ep450",
        "checkpoint_ep500",
        "best",
    ]

    target = 72.111
    best_match = None
    best_diff  = 999

    print(f"{'Checkpoint':<22} "
          f"{'CSR (%)':>10} "
          f"{'Diff from 72.111':>18}")
    print("-" * 54)

    for ckpt in checkpoints:
        path = os.path.join(base, ckpt)
        if not os.path.exists(path):
            print(f"  {ckpt:<20} MISSING")
            continue
        csr, status = quick_eval(
            path, workload)
        if status != "OK":
            print(f"  {ckpt:<20} ERROR: {status}")
            continue
        diff = abs(csr - target)
        marker = " ← BEST" \
            if diff < best_diff else ""
        if diff < best_diff:
            best_diff  = diff
            best_match = ckpt
        print(f"  {ckpt:<20} "
              f"{csr:>10.3f} "
              f"{diff:>16.3f}{marker}")

    print("\n" + "=" * 55)
    print(f"Best matching checkpoint: "
          f"{best_match}")
    print(f"Difference from 72.111%: "
          f"{best_diff:.3f} pp")
    print("=" * 55)
    print(f"\nTo restore, run:")
    print(f'Copy-Item -Path '
          f'"trained_model_tascar\\'
          f'{best_match}\\*" '
          f'-Destination '
          f'"trained_model_tascar\\best\\" '
          f'-Force')


if __name__ == "__main__":
    main()
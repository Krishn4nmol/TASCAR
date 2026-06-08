# check_checkpoint.py
# Complete checkpoint comparison
# FIXED: handle_request FIRST
# then scaling decision!

import numpy as np
from collections import Counter
from config import (
    NUM_QUEUES,
    NUM_FUNCTIONS,
    EVAL_CALLS,
    DELTA,
    TASCAR_EVAL_DELTA,
    TRANSFORMER_DIM,
    SEQUENCE_LENGTH,
    MODEL_SAVE_PATH,
    SCALING_FACTOR)
from simulator import AzureDataLoader
from metrics_tracker import MetricsTracker
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent
from ppo_agent import PPOAgent

TEST_CALLS = 100000


# ─────────────────────────────────────────
# LOAD WORKLOAD
# EXACT same as evaluate_tascar.py!
# ─────────────────────────────────────────

def load_common_workload():
    """
    Loads Common workload exactly
    as evaluate_tascar.py does!
    Same seed! Same selection!
    """
    print("Loading Common workload...")
    loader = AzureDataLoader()
    day1   = loader.load_day(1)
    counts = Counter(
        c.function_id for c in day1)
    top = set(
        f for f, _ in
        counts.most_common(NUM_FUNCTIONS))
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
    print(f"  {len(common)} calls!")
    return common


# ─────────────────────────────────────────
# TEST CASR
# FIXED: handle THEN scale!
# ─────────────────────────────────────────

def test_casr(test_calls):
    """Tests CASR with PPO agent"""
    import os
    print("\nTesting CASR...")

    model_path = MODEL_SAVE_PATH + "best/"
    if not os.path.exists(
            model_path + "actor.pth"):
        print("  No CASR model!")
        return None, None

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES
    agent      = PPOAgent(
        state_dim, action_dim)
    agent.load(model_path)

    action_map = {}
    choices    = [
        -SCALING_FACTOR, 0,
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

    for call in test_calls:
        # FIXED: handle FIRST!
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
            for q_idx, scale in enumerate(
                    action_map[action]):
                if scale != 0:
                    tracker.scale_queue(
                        q_idx, scale)

    metrics = tracker.get_all_metrics()
    csr     = metrics['cold_start_rate']
    print(f"  CSR: {csr:.3f}%")
    print(f"  WMT: "
          f"{metrics['avg_wasted_memory_time']:.3f}s")
    print(f"  TPI: {metrics['tpi']:.2f}")
    print(f"  ART: "
          f"{metrics['avg_response_time']:.3f}s")
    print(f"  SVR: "
          f"{metrics['sla_violation_rate']:.2f}%")
    return csr, metrics


# ─────────────────────────────────────────
# TEST TASCAR CHECKPOINT
# FIXED: handle THEN scale!
# ─────────────────────────────────────────

def test_tascar(test_calls,
                model_path, name):
    """Tests TASCAR checkpoint"""
    import os
    print(f"\nTesting {name}...")

    if not os.path.exists(
            model_path + "actor.pth"):
        print(f"  No model found!")
        return None, None

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

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

    for call in test_calls:
        # FIXED: handle FIRST!
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
            enc = agent.get_encoded_state(
                seq)
            action = agent.choose_action(
                enc, evaluate=True)
            for q_idx, scale in enumerate(
                    agent.action_map[
                        action]):
                if scale != 0:
                    tracker.scale_queue(
                        q_idx, scale)

    metrics = tracker.get_all_metrics()
    csr     = metrics['cold_start_rate']
    print(f"  CSR: {csr:.3f}%")
    print(f"  WMT: "
          f"{metrics['avg_wasted_memory_time']:.3f}s")
    print(f"  TPI: {metrics['tpi']:.2f}")
    print(f"  ART: "
          f"{metrics['avg_response_time']:.3f}s")
    print(f"  SVR: "
          f"{metrics['sla_violation_rate']:.2f}%")
    return csr, metrics


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

print("=" * 55)
print("Checkpoint Comparison")
print("FIXED: handle THEN scale!")
print(f"Calls: {TEST_CALLS}")
print("=" * 55)

# Load EXACT same workload!
test_calls = load_common_workload()

# Test CASR
casr_csr, casr_metrics = test_casr(
    test_calls)

# Test all checkpoints
checkpoints = [
    ('trained_model_tascar/best/',
     'Best Model'),
    ('trained_model_tascar/checkpoint_ep500/',
     'checkpoint_ep500'),
    ('trained_model_tascar/checkpoint_ep450/',
     'checkpoint_ep450'),
    ('trained_model_tascar/checkpoint_ep400/',
     'checkpoint_ep400'),
    ('trained_model_tascar/checkpoint_ep350/',
     'checkpoint_ep350'),
    ('trained_model_tascar/checkpoint_ep300/',
     'checkpoint_ep300'),
    ('trained_model_tascar/checkpoint_ep250/',
     'checkpoint_ep250'),
]

results = {}
for path, name in checkpoints:
    csr, metrics = test_tascar(
        test_calls, path, name)
    if csr is not None:
        results[name] = {
            'csr':     csr,
            'metrics': metrics}

# Summary
print("\n" + "=" * 60)
print("SUMMARY (handle THEN scale)")
print("=" * 60)
print(
    f"\n{'Algorithm':<25}"
    f"{'CSR':>8}"
    f"{'TPI':>8}"
    f"{'vs CASR':>14}")
print("-" * 58)

if casr_csr:
    print(
        f"{'CASR':<25}"
        f"{casr_csr:>8.3f}%"
        f"{casr_metrics['tpi']:>8.2f}"
        f"{'baseline':>14}")

for name, data in results.items():
    csr  = data['csr']
    tpi  = data['metrics']['tpi']
    diff = casr_csr - csr
    if diff > 0:
        result = f"+{diff:.3f}pp ✅"
    else:
        result = f"{diff:.3f}pp ❌"
    print(
        f"{name:<25}"
        f"{csr:>8.3f}%"
        f"{tpi:>8.2f}"
        f"{result:>14}")

print("\n" + "=" * 60)
if results:
    best = min(
        results,
        key=lambda x: results[x]['csr'])
    best_csr = results[best]['csr']
    diff     = casr_csr - best_csr
    print(f"\nBest checkpoint: {best}")
    print(f"CSR:  {best_csr:.3f}%")
    if diff > 0:
        print(
            f"BEATS CASR by "
            f"{diff:.3f}pp! ✅")
    else:
        print(
            f"{abs(diff):.3f}pp "
            f"worse than CASR! ❌")
    print(
        f"\nTo use this checkpoint:")
    best_path = [
        p for p, n in checkpoints
        if n == best][0]
    print(
        f"Copy-Item "
        f"{best_path}* "
        f"trained_model_tascar\\best\\ "
        f"-Force")
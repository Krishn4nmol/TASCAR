# train_casr200_seed456.py
# Retrains CASR for 200 episodes with seed=456
# Multi-seed ablation companion to V1 (seed 42)
# Same architecture as original CASR (PPO, single state, theta=0.8)

import numpy as np
import json
import os
import time
from collections import Counter

from config import (
    NUM_QUEUES,
    TRAIN_DAYS,
    NUM_FUNCTIONS,
    EVAL_CALLS,
    DELTA,
    THETA,
    SCALING_FACTOR,
    LEARNING_RATE_ACTOR,
    LEARNING_RATE_CRITIC,
    DISCOUNT_FACTOR,
    GAE_LAMBDA,
    PPO_CLIP,
    ENTROPY_COEFF,
    MINI_BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    EPOCHS_PER_UPDATE,
    CONVERGENCE_WINDOW,
    CONVERGENCE_THRESHOLD,
    RANDOM_SEED,
    ABLATION_RESULTS)
from simulator import AzureDataLoader
from scache import SCache
from ppo_agent import PPOAgent

# Save path for seed-456 CASR-200 retrain
CASR_500_PATH = "trained_model_casr200_seed456/"


# ─────────────────────────────────────────
# LOAD DATA
# Same as original CASR training!
# ─────────────────────────────────────────

def load_filtered_data():
    loader     = AzureDataLoader()
    train_data = []
    for day in TRAIN_DAYS:
        print(f"  Loading day {day}...")
        day_calls = loader.load_day(day)
        train_data.extend(day_calls)
    func_counts   = Counter(
        c.function_id
        for c in train_data)
    top_functions = set(
        f for f, _ in
        func_counts.most_common(
            NUM_FUNCTIONS))
    train_data = [
        c for c in train_data
        if c.function_id
        in top_functions]
    print(
        f"  Total: {len(train_data)}")
    return train_data


# ─────────────────────────────────────────
# NORMALIZE STATE
# ─────────────────────────────────────────

def normalize_state(raw_state):
    state = np.array(
        raw_state, dtype=np.float32)
    if np.isnan(state).any():
        return np.zeros_like(state)
    mean = np.mean(state)
    std  = np.std(state)
    if std > 0:
        state = (state - mean) / std
    if np.isnan(state).any():
        return np.zeros(
            len(raw_state),
            dtype=np.float32)
    return state


# ─────────────────────────────────────────
# REWARD NORMALIZER
# Same as original CASR!
# ─────────────────────────────────────────

class RewardNormalizer:
    def __init__(self):
        self.r1_min =  float('inf')
        self.r1_max = -float('inf')
        self.r2_min =  float('inf')
        self.r2_max = -float('inf')

    def calculate(self,
                  cold_starts,
                  wmt_change,
                  theta=THETA):
        r1 = float(cold_starts)
        r2 = float(max(0, wmt_change))
        if r1 < self.r1_min:
            self.r1_min = r1
        if r1 > self.r1_max:
            self.r1_max = r1
        if r2 < self.r2_min:
            self.r2_min = r2
        if r2 > self.r2_max:
            self.r2_max = r2
        r1_range = (
            self.r1_max - self.r1_min)
        r1_norm  = (
            (r1 - self.r1_min) /
            r1_range
            if r1_range > 0 else 0.0)
        r2_range = (
            self.r2_max - self.r2_min)
        r2_norm  = (
            (r2 - self.r2_min) /
            r2_range
            if r2_range > 0 else 0.0)
        reward = -(
            theta * r1_norm +
            (1 - theta) * r2_norm)
        if np.isnan(reward):
            reward = 0.0
        return float(reward)


# ─────────────────────────────────────────
# MAIN TRAINING
# CASR architecture exactly!
# PPO + single state!
# Fixed theta = 0.8!
# 200 episodes, seed 456!
# ─────────────────────────────────────────

def train_casr_500():
    """
    Train CASR for 200 episodes, seed 456!
    Exact same architecture as original V1!
    PPO + single state snapshot!
    Fixed theta = 0.8!
    Multi-seed ablation companion run.
    """
    os.makedirs(
        CASR_500_PATH, exist_ok=True)
    os.makedirs(
        ABLATION_RESULTS,
        exist_ok=True)

    RANDOM_SEED = 456
    np.random.seed(RANDOM_SEED)
    print("CASR 200 Episode Training!")
    print(
        f"Seed: {RANDOM_SEED}")
    print(
        "Same architecture as original!")
    print(
        "PPO + single state + theta=0.8!")
    print(
        "200 episodes, multi-seed ablation!")

    print("\nLoading dataset...")
    train_data = load_filtered_data()

    state_dim    = NUM_QUEUES * 7
    action_dim   = 3 ** NUM_QUEUES
    calls_per_ep = EVAL_CALLS

    print(f"\nCASR-200 (seed 456) Configuration:")
    print(f"  State dim:     {state_dim}")
    print(f"  Action dim:    {action_dim}")
    print(f"  Delta:         {DELTA}")
    print(
        f"  Steps/episode: "
        f"{calls_per_ep // DELTA}")
    print(f"  Episodes:      200")
    print(f"  Theta:         {THETA} (fixed!)")
    print(f"  Algorithm:     PPO")
    print(f"  NO Transformer!")
    print(f"  NO SAC!")

    agent = PPOAgent(
        state_dim, action_dim)

    # Build action map
    action_map = {}
    choices    = [
        -SCALING_FACTOR,
        0,
        SCALING_FACTOR]
    for i in range(
            3 ** NUM_QUEUES):
        action = []
        temp   = i
        for _ in range(NUM_QUEUES):
            action.append(
                choices[temp % 3])
            temp //= 3
        action_map[i] = action

    reward_norm  = RewardNormalizer()
    best_reward  = float('-inf')
    start_time   = time.time()

    rewards_log    = []
    cold_rates_log = []
    episodes_log   = []
    convergence_ep = -1

    print(
        "\nStarting CASR-200 (seed 456) training...")
    print("=" * 50)

    for episode in range(1, 201):

        max_start = max(
            1,
            len(train_data) -
            calls_per_ep)
        start_idx = np.random.randint(
            0, max_start)
        episode_calls = train_data[
            start_idx:
            start_idx + calls_per_ep]
        if len(episode_calls) < 1000:
            episode_calls = (
                train_data[:calls_per_ep])

        scache    = SCache()
        raw_state = normalize_state(
            scache.get_state())

        ep_reward  = 0.0
        ep_cold    = 0
        ep_warm    = 0
        step_cold  = 0
        step_warm  = 0
        call_count = 0
        wmt_before = 0.0
        steps_done = 0

        for call in episode_calls:
            is_warm = (
                scache.handle_request(
                    call))
            if is_warm:
                step_warm += 1
                ep_warm   += 1
            else:
                step_cold += 1
                ep_cold   += 1
            call_count += 1

            if call_count % DELTA == 0:
                new_raw = normalize_state(
                    scache.get_state())

                total = (
                    step_cold + step_warm)
                cold_rate = (
                    step_cold / total
                    if total > 0 else 0)

                current_wmt = (
                    scache
                    .get_total_wasted_memory_time())
                wmt_change = max(
                    0,
                    current_wmt -
                    wmt_before)
                wmt_before = current_wmt

                reward = (
                    reward_norm
                    .calculate(
                        step_cold,
                        wmt_change,
                        THETA))

                action, log_prob = (
                    agent.choose_action(
                        raw_state))

                agent.store_experience(
                    raw_state,
                    action,
                    log_prob,
                    reward,
                    new_raw,
                    False)

                ep_reward  += reward
                steps_done += 1

                # PPO update when ready
                if agent.buffer.is_ready():
                    agent.update()

                scales = action_map[
                    action]
                for q_idx, scale in (
                        enumerate(
                            scales)):
                    if scale != 0:
                        scache.scale_queue(
                            q_idx, scale)

                raw_state = new_raw
                step_cold = 0
                step_warm = 0

        total_calls = ep_cold + ep_warm
        cold_pct = (
            ep_cold / total_calls * 100
            if total_calls > 0 else 0)

        avg_ep_reward = (
            ep_reward / steps_done
            if steps_done > 0
            else ep_reward)

        rewards_log.append(avg_ep_reward)
        cold_rates_log.append(cold_pct)
        episodes_log.append(episode)

        if avg_ep_reward > best_reward:
            best_reward = avg_ep_reward
            agent.save(
                CASR_500_PATH + "best/")

        if episode % 50 == 0:
            agent.save(
                CASR_500_PATH +
                f"checkpoint_ep"
                f"{episode}/")

        if (convergence_ep == -1 and
                len(rewards_log) >=
                CONVERGENCE_WINDOW):
            recent = rewards_log[
                -CONVERGENCE_WINDOW:]
            if np.std(recent) < (
                    CONVERGENCE_THRESHOLD):
                convergence_ep = episode

        if episode % 10 == 0:
            avg_r = np.mean(
                rewards_log[-10:])
            avg_c = np.mean(
                cold_rates_log[-10:])
            elapsed = (
                time.time() - start_time)
            print(
                f"Ep {episode:3d} | "
                f"Reward: {avg_r:7.4f} | "
                f"Cold%: {avg_c:5.1f}% | "
                f"Time: {elapsed:.0f}s")

    elapsed = (
        time.time() - start_time)
    agent.save(
        CASR_500_PATH + "best/")

    logs = {
        'variant':
            'V1_CASR_seed456',
        'episodes':
            episodes_log,
        'rewards':
            rewards_log,
        'cold_start_rates':
            cold_rates_log,
        'best_reward':
            best_reward,
        'training_time':
            elapsed,
        'convergence_ep':
            convergence_ep,
        'random_seed':
            RANDOM_SEED,
        'has_transformer': False,
        'has_sac':         False,
        'has_dynamic_theta': False,
        'episodes_trained': 200,
    }
    log_path = (
        ABLATION_RESULTS +
        'casr200_seed456_logs.json')
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

    print("\n" + "=" * 50)
    print("CASR-200 (seed 456) Training Complete!")
    print("=" * 50)
    print(
        f"Best reward:    "
        f"{best_reward:.4f}")
    print(
        f"Training time:  "
        f"{elapsed:.1f}s")
    print(
        f"Convergence ep: "
        f"{convergence_ep}")
    print(
        f"Saved to:       "
        f"{CASR_500_PATH}best/")

    return agent


# ─────────────────────────────────────────
# QUICK EVAL AFTER TRAINING
# ─────────────────────────────────────────

def quick_check():
    """
    Quick check: CASR-200 (seed 456) CSR
    on Common workload.
    """
    from metrics_tracker import (
        MetricsTracker)
    from simulator import AzureDataLoader
    import numpy as np

    print("\n" + "=" * 50)
    print("Quick Check: CASR-200 seed 456")
    print("=" * 50)

    QUICK_CHECK_SEED = 456

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
    np.random.seed(QUICK_CHECK_SEED)
    if len(common) > EVAL_CALLS:
        idx = np.random.choice(
            len(common), EVAL_CALLS,
            replace=False)
        idx.sort()
        common = [
            common[i] for i in idx]

    model_path = CASR_500_PATH + "best/"
    if not os.path.exists(
            model_path + "actor.pth"):
        print("No model found!")
        return

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

    for call in common:
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

    metrics = tracker.get_all_metrics()
    csr     = metrics['cold_start_rate']

    print(
        f"\nCASR-200 (seed 456) CSR: {csr:.3f}%")
    print(
        f"Compare against V1 (seed 42): 90.930%")


if __name__ == "__main__":
    print("=" * 50)
    print("CASR-200 Episode Training")
    print("Multi-seed ablation: seed 456")
    print("PPO only! No Transformer!")
    print("Seed: 456")
    print("=" * 50)
    train_casr_500()
    quick_check()
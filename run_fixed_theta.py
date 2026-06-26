# run_fixed_theta.py
# Trains TASCAR with FIXED theta=0.7
# for dynamic theta ablation study
# Compares: dynamic theta vs fixed theta
# Seeds: 42, 123, 456
# Saves to trained_model_fixed_theta_seedXXX/

import numpy as np
import json
import os
import time
from collections import Counter

from config import (
    NUM_QUEUES, NUM_FUNCTIONS,
    EVAL_CALLS, DELTA,
    TASCAR_EVAL_DELTA,
    SEQUENCE_LENGTH, TRANSFORMER_DIM,
    TASCAR_EPISODES, TASCAR_DELTA,
    SAC_UPDATES_PER_STEP,
    SCALING_FACTOR, MODEL_SAVE_PATH)
from simulator import AzureDataLoader
from metrics_tracker import MetricsTracker
from transformer_encoder import (
    TransformerEncoder, StateHistoryBuffer)
from sac_agent import SACAgent
from ppo_agent import PPOAgent
from train_tascar import (
    load_filtered_data, normalize_state,
    RewardNormalizer, TASCARLogger,
    warmup_buffer)
from scache import SCache

# ─────────────────────────────────────────
# FIXED THETA CONFIG
# θ=0.7 is midpoint of [0.5, 0.9]
# Dynamic TASCAR adapts θ in this range
# Fixed ablation uses constant θ=0.7
# ─────────────────────────────────────────
FIXED_THETA = 0.7
SEEDS = [42, 123, 456]
RESULTS_PATH = "results_fixed_theta/"


def train_fixed_theta(seed, model_path,
                      results_path):
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    np.random.seed(seed)
    print(f"\nTraining seed {seed} "
          f"(fixed θ={FIXED_THETA})...")

    train_data = load_filtered_data()

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    transformer = TransformerEncoder(state_dim)
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=transformer)

    data_rng = np.random.RandomState(seed)
    warmup_buffer(
        agent, train_data, state_dim,
        data_rng, warmup_episodes=20)

    reward_norm = RewardNormalizer()
    logger      = TASCARLogger()
    logger.start_training()

    print(f"  Training {TASCAR_EPISODES} "
          f"episodes with fixed θ={FIXED_THETA}...")

    for episode in range(1, TASCAR_EPISODES + 1):
        max_start = max(
            1, len(train_data) - EVAL_CALLS)
        start_idx = data_rng.randint(0, max_start)
        episode_calls = train_data[
            start_idx:start_idx + EVAL_CALLS]
        if len(episode_calls) < 1000:
            episode_calls = train_data[:EVAL_CALLS]

        scache  = SCache()
        history = StateHistoryBuffer(
            SEQUENCE_LENGTH, state_dim)
        raw = normalize_state(scache.get_state())
        history.add(raw)
        encoded_state = agent.get_encoded_state(
            history.get_sequence())

        ep_reward = ep_cold = ep_warm = 0.0
        step_cold = step_warm = 0
        call_count = steps_done = 0
        wmt_before = 0.0

        for call in episode_calls:
            is_warm = scache.handle_request(call)
            if is_warm: step_warm += 1; ep_warm += 1
            else:       step_cold += 1; ep_cold += 1
            call_count += 1

            if call_count % TASCAR_DELTA == 0:
                new_raw = normalize_state(
                    scache.get_state())
                history.add(new_raw)
                next_encoded = agent.get_encoded_state(
                    history.get_sequence())

                current_wmt = (
                    scache.get_total_wasted_memory_time())
                wmt_change = max(
                    0, current_wmt - wmt_before)
                wmt_before = current_wmt

                # FIXED THETA — no adaptation!
                reward = reward_norm.calculate(
                    step_cold, wmt_change, FIXED_THETA)
                action = agent.choose_action(encoded_state)

                if (not np.isnan(encoded_state).any() and
                        not np.isnan(next_encoded).any()):
                    agent.store_experience(
                        encoded_state, action,
                        reward, next_encoded, False)

                ep_reward  += reward
                steps_done += 1

                for _ in range(SAC_UPDATES_PER_STEP):
                    agent.update()

                scales = agent.action_map[action]
                for q_idx, scale in enumerate(scales):
                    if scale != 0:
                        scache.scale_queue(q_idx, scale)

                encoded_state = next_encoded
                step_cold = step_warm = 0

        total_calls = ep_cold + ep_warm
        cold_pct = (ep_cold / total_calls * 100
                    if total_calls > 0 else 0)
        avg_reward = (ep_reward / steps_done
                      if steps_done > 0 else ep_reward)

        logger.log_episode(
            episode, avg_reward, cold_pct,
            scache.get_total_wasted_memory_time(),
            FIXED_THETA, steps_this_ep=steps_done)

        if avg_reward > logger.best_reward:
            agent.save(model_path + "best/")

        if episode % 50 == 0:
            agent.save(
                model_path + f"checkpoint_ep{episode}/")

        if episode % 10 == 0:
            avg_r = np.mean(logger.rewards[-10:])
            avg_c = np.mean(
                logger.cold_start_rates[-10:])
            print(f"  Ep {episode:3d} | "
                  f"Reward: {avg_r:7.4f} | "
                  f"Cold%: {avg_c:5.1f}% | "
                  f"θ=FIXED({FIXED_THETA}) | "
                  f"Time: {logger.get_training_time():.0f}s")

    logger.end_training()
    agent.save(model_path + "best/")
    logger.save_logs(results_path)

    print(f"\n  Seed {seed} done!")
    print(f"  Best reward: {logger.best_reward:.4f}")
    print(f"  Time: {logger.get_training_time():.1f}s")


def find_best_checkpoint(model_path, workload):
    """Find best checkpoint by CSR — same as multiseed"""
    import glob, shutil
    print(f"  Finding best checkpoint...")

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES
    checkpoints = sorted(glob.glob(
        model_path + "checkpoint_ep*/"))

    best_csr  = 999.0
    best_path = model_path + "best/"

    for ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path + "actor.pth"):
            continue
        try:
            transformer = TransformerEncoder(state_dim)
            agent = SACAgent(
                transformer_dim=TRANSFORMER_DIM,
                action_dim=action_dim,
                transformer=transformer)
            agent.load(ckpt_path)

            tracker    = MetricsTracker()
            history    = StateHistoryBuffer(
                SEQUENCE_LENGTH, state_dim)
            call_count = 0

            for call in workload:
                tracker.handle_request(call)
                call_count += 1
                if call_count % TASCAR_EVAL_DELTA == 0:
                    raw = np.array(
                        tracker.get_state(),
                        dtype=np.float32)
                    mean = np.mean(raw)
                    std  = np.std(raw)
                    if std > 0:
                        raw = (raw - mean) / std
                    history.add(raw)
                    enc = agent.get_encoded_state(
                        history.get_sequence())
                    act = agent.choose_action(
                        enc, evaluate=True)
                    for q_idx, scale in enumerate(
                            agent.action_map[act]):
                        if scale != 0:
                            tracker.scale_queue(
                                q_idx, scale)

            csr = tracker.get_all_metrics()[
                'cold_start_rate']
            ckpt_name = os.path.basename(
                ckpt_path.rstrip('/'))
            print(f"    {ckpt_name}: CSR={csr:.3f}%")

            if csr < best_csr:
                best_csr  = csr
                best_path = ckpt_path

        except Exception as e:
            print(f"    Error: {e}")
            continue

    print(f"  Best: {best_path} CSR={best_csr:.3f}%")
    best_dir = model_path + "best/"
    os.makedirs(best_dir, exist_ok=True)
    import shutil
    for f in os.listdir(best_path):
        shutil.copy2(
            os.path.join(best_path, f),
            os.path.join(best_dir, f))
    return best_csr


def evaluate_fixed_theta(seed, model_path,
                         workloads):
    print(f"\nEvaluating seed {seed}...")

    find_best_checkpoint(
        model_path, workloads['Common'])

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    casr_path  = MODEL_SAVE_PATH + "best/"
    casr_agent = PPOAgent(state_dim, action_dim)
    action_map = {}
    choices    = [-SCALING_FACTOR, 0, SCALING_FACTOR]
    for i in range(3 ** NUM_QUEUES):
        action = []
        temp   = i
        for _ in range(NUM_QUEUES):
            action.append(choices[temp % 3])
            temp //= 3
        action_map[i] = action
    if os.path.exists(casr_path + "actor.pth"):
        casr_agent.load(casr_path)

    best_path = model_path + "best/"
    transformer = TransformerEncoder(state_dim)
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=transformer)
    agent.load(best_path)

    seed_results = {}

    for wl_name, calls in workloads.items():
        print(f"  Workload: {wl_name}")

        # CASR
        casr_tracker = MetricsTracker()
        call_count   = 0
        for call in calls:
            casr_tracker.handle_request(call)
            call_count += 1
            if call_count % DELTA == 0:
                state = np.array(
                    casr_tracker.get_state(),
                    dtype=np.float32)
                mean = np.mean(state)
                std  = np.std(state)
                if std > 0:
                    state = (state - mean) / std
                act, _ = casr_agent.choose_action(state)
                for q_idx, scale in enumerate(
                        action_map[act]):
                    if scale != 0:
                        casr_tracker.scale_queue(
                            q_idx, scale)
        casr_m = casr_tracker.get_all_metrics()

        # Fixed theta TASCAR
        fixed_tracker = MetricsTracker()
        history       = StateHistoryBuffer(
            SEQUENCE_LENGTH, state_dim)
        call_count    = 0
        for call in calls:
            fixed_tracker.handle_request(call)
            call_count += 1
            if call_count % TASCAR_EVAL_DELTA == 0:
                raw = np.array(
                    fixed_tracker.get_state(),
                    dtype=np.float32)
                mean = np.mean(raw)
                std  = np.std(raw)
                if std > 0:
                    raw = (raw - mean) / std
                history.add(raw)
                enc = agent.get_encoded_state(
                    history.get_sequence())
                act = agent.choose_action(
                    enc, evaluate=True)
                for q_idx, scale in enumerate(
                        agent.action_map[act]):
                    if scale != 0:
                        fixed_tracker.scale_queue(
                            q_idx, scale)
        fixed_m = fixed_tracker.get_all_metrics()

        casr_csr  = casr_m['cold_start_rate']
        fixed_csr = fixed_m['cold_start_rate']

        seed_results[wl_name] = {
            'CASR':        casr_csr,
            'Fixed_Theta': fixed_csr,
        }

        print(f"    CASR:        {casr_csr:.3f}%")
        print(f"    Fixed θ=0.7: {fixed_csr:.3f}%")

    return seed_results


def load_workloads():
    loader    = AzureDataLoader()
    workloads = {}
    print("\nPreparing workloads...")

    day1   = loader.load_day(1)
    counts = Counter(c.function_id for c in day1)
    top = set(f for f, _ in
              counts.most_common(NUM_FUNCTIONS))
    common = [c for c in day1
              if c.function_id in top]
    np.random.seed(42)
    if len(common) > EVAL_CALLS:
        idx = np.random.choice(
            len(common), EVAL_CALLS, replace=False)
        idx.sort()
        common = [common[i] for i in idx]
    workloads['Common'] = common
    print(f"  Common: {len(common)} calls")

    day2  = loader.load_day(2)
    heavy = [c for c in day2
             if c.cold_start_overhead > 1]
    counts = Counter(c.function_id for c in heavy)
    top = set(f for f, _ in
              counts.most_common(NUM_FUNCTIONS))
    significant = [c for c in heavy
                   if c.function_id in top]
    np.random.seed(42)
    if len(significant) > EVAL_CALLS:
        idx = np.random.choice(
            len(significant), EVAL_CALLS,
            replace=False)
        idx.sort()
        significant = [significant[i] for i in idx]
    workloads['Significant'] = significant
    print(f"  Significant: {len(significant)} calls")

    day3  = loader.load_day(3)
    funcs = list(set(c.function_id for c in day3))
    np.random.seed(43)
    np.random.shuffle(funcs)
    selected  = set(funcs[:NUM_FUNCTIONS])
    random_wl = [c for c in day3
                 if c.function_id in selected]
    np.random.seed(43)
    if len(random_wl) > EVAL_CALLS:
        idx = np.random.choice(
            len(random_wl), EVAL_CALLS,
            replace=False)
        idx.sort()
        random_wl = [random_wl[i] for i in idx]
    workloads['Random'] = random_wl
    print(f"  Random: {len(random_wl)} calls")

    return workloads


if __name__ == "__main__":
    print("=" * 60)
    print("Dynamic θ Ablation Study")
    print(f"Fixed θ={FIXED_THETA} vs Dynamic θ")
    print(f"Seeds: {SEEDS}")
    print("~4 hours total")
    print("=" * 60)

    workloads   = load_workloads()
    all_results = {}

    for seed in SEEDS:
        model_path   = (f"trained_model_fixed_theta"
                        f"_seed{seed}/")
        results_path = (f"results_fixed_theta"
                        f"_seed{seed}/")

        print(f"\n{'='*50}")
        print(f"Seed: {seed}")
        print(f"{'='*50}")

        train_fixed_theta(seed, model_path,
                          results_path)
        seed_results = evaluate_fixed_theta(
            seed, model_path, workloads)

        if seed_results:
            all_results[seed] = seed_results

        time.sleep(10)

    # Summary
    print("\n" + "=" * 60)
    print("DYNAMIC θ ABLATION RESULTS")
    print("Fixed θ=0.7 vs Dynamic θ (TASCAR)")
    print("=" * 60)

    # Dynamic TASCAR reference values from paper
    dynamic_ref = {
        'Common':      71.860,
        'Significant': 74.555,
        'Random':      70.844,
    }

    for wl in ['Common', 'Significant', 'Random']:
        fixed_vals = [
            all_results[s][wl]['Fixed_Theta']
            for s in SEEDS
            if s in all_results]
        if fixed_vals:
            mean_fixed = np.mean(fixed_vals)
            std_fixed  = np.std(fixed_vals)
            dynamic    = dynamic_ref[wl]
            diff       = mean_fixed - dynamic
            print(f"\n{wl}:")
            print(f"  Fixed θ=0.7: "
                  f"{mean_fixed:.3f}±{std_fixed:.3f}%")
            print(f"  Dynamic θ:   {dynamic:.3f}% "
                  f"(from Table V)")
            symbol = "✅ Dynamic better" if diff > 0 \
                else "❌ Fixed better"
            print(f"  Difference:  {diff:+.3f}pp "
                  f"{symbol}")

    # Save
    os.makedirs(RESULTS_PATH, exist_ok=True)
    with open(RESULTS_PATH + 'fixed_theta_results.json',
              'w') as f:
        json.dump(
            {str(k): v for k, v in all_results.items()},
            f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}fixed_theta_results.json")
    print("=" * 60)
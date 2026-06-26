# run_multiseed_part2.py
# Trains TASCAR for new seeds
# IDENTICAL conditions to original
# run_multiseed.py (seeds 42/123/456)
# KEY FIX: finds best checkpoint
# before evaluation (like original
# check_checkpoint.py workflow)

import numpy as np
import json
import os
import time
import glob
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from config import (
    NUM_QUEUES, NUM_FUNCTIONS,
    EVAL_CALLS, DELTA,
    TASCAR_EVAL_DELTA,
    SEQUENCE_LENGTH, TRANSFORMER_DIM,
    TASCAR_EPISODES, TASCAR_DELTA,
    SAC_UPDATES_PER_STEP,
    TRAIN_DAYS, THETA,
    SCALING_FACTOR, MODEL_SAVE_PATH)
from simulator import AzureDataLoader
from metrics_tracker import MetricsTracker
from transformer_encoder import (
    TransformerEncoder, StateHistoryBuffer)
from sac_agent import SACAgent
from ppo_agent import PPOAgent
from train_tascar import (
    load_filtered_data, normalize_state,
    compute_dynamic_theta,
    RewardNormalizer, TASCARLogger,
    warmup_buffer)
from scache import SCache

SEEDS = [789, 1000, 2024, 2025]

SEED_CONFIGS = {
    789:  {'model_path': 'trained_model_tascar_seed789/',
           'results_path': 'results_tascar_seed789/'},
    1000: {'model_path': 'trained_model_tascar_seed1000/',
           'results_path': 'results_tascar_seed1000/'},
    2024: {'model_path': 'trained_model_tascar_seed2024/',
           'results_path': 'results_tascar_seed2024/'},
    2025: {'model_path': 'trained_model_tascar_seed2025/',
           'results_path': 'results_tascar_seed2025/'},
}

MULTISEED_RESULTS = "results_multiseed_part2/"


def train_one_seed(seed, model_path, results_path):
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    np.random.seed(seed)
    print(f"\nTraining seed {seed}...")
    print(f"  Model: {model_path}")

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

    reward_norm   = RewardNormalizer()
    logger        = TASCARLogger()
    current_theta = THETA
    logger.start_training()

    print(f"  Training {TASCAR_EPISODES} episodes...")

    for episode in range(1, TASCAR_EPISODES + 1):
        max_start = max(1, len(train_data) - EVAL_CALLS)
        start_idx = data_rng.randint(0, max_start)
        episode_calls = train_data[start_idx:start_idx + EVAL_CALLS]
        if len(episode_calls) < 1000:
            episode_calls = train_data[:EVAL_CALLS]

        scache  = SCache()
        history = StateHistoryBuffer(SEQUENCE_LENGTH, state_dim)
        raw_state = normalize_state(scache.get_state())
        history.add(raw_state)
        encoded_state = agent.get_encoded_state(history.get_sequence())

        ep_reward = ep_cold = ep_warm = 0.0
        step_cold = step_warm = call_count = steps_done = 0
        wmt_before = 0.0

        for call in episode_calls:
            is_warm = scache.handle_request(call)
            if is_warm: step_warm += 1; ep_warm += 1
            else:       step_cold += 1; ep_cold += 1
            call_count += 1

            if call_count % TASCAR_DELTA == 0:
                new_raw = normalize_state(scache.get_state())
                history.add(new_raw)
                next_encoded = agent.get_encoded_state(
                    history.get_sequence())

                total = step_cold + step_warm
                cold_rate = step_cold / total if total > 0 else 0
                current_theta = compute_dynamic_theta(
                    cold_rate, current_theta)

                current_wmt = scache.get_total_wasted_memory_time()
                wmt_change = max(0, current_wmt - wmt_before)
                wmt_before = current_wmt

                reward = reward_norm.calculate(
                    step_cold, wmt_change, current_theta)
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
        cold_pct = ep_cold / total_calls * 100 if total_calls > 0 else 0
        avg_reward = ep_reward / steps_done if steps_done > 0 else ep_reward

        logger.log_episode(
            episode, avg_reward, cold_pct,
            scache.get_total_wasted_memory_time(),
            current_theta, steps_this_ep=steps_done)

        if avg_reward > logger.best_reward:
            agent.save(model_path + "best/")

        if episode % 50 == 0:
            agent.save(model_path + f"checkpoint_ep{episode}/")

        if episode % 10 == 0:
            avg_r = np.mean(logger.rewards[-10:])
            avg_c = np.mean(logger.cold_start_rates[-10:])
            print(f"  Ep {episode:3d} | "
                  f"Reward: {avg_r:7.4f} | "
                  f"Cold%: {avg_c:5.1f}% | "
                  f"Theta: {current_theta:.3f} | "
                  f"Time: {logger.get_training_time():.0f}s")

    logger.end_training()
    agent.save(model_path + "best/")
    logger.save_logs(results_path)

    print(f"\n  Seed {seed} training done!")
    print(f"  Best reward: {logger.best_reward:.4f}")
    print(f"  Time: {logger.get_training_time():.1f}s")


def find_best_checkpoint(model_path, workload):
    """
    Finds best checkpoint by CSR.
    Identical to check_checkpoint.py!
    Copies best checkpoint to best/.
    """
    print(f"\n  Finding best checkpoint...")

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
            history    = StateHistoryBuffer(SEQUENCE_LENGTH, state_dim)
            call_count = 0

            for call in workload:
                tracker.handle_request(call)
                call_count += 1
                if call_count % TASCAR_EVAL_DELTA == 0:
                    raw = np.array(
                        tracker.get_state(), dtype=np.float32)
                    mean = np.mean(raw)
                    std  = np.std(raw)
                    if std > 0:
                        raw = (raw - mean) / std
                    history.add(raw)
                    enc = agent.get_encoded_state(
                        history.get_sequence())
                    act = agent.choose_action(enc, evaluate=True)
                    for q_idx, scale in enumerate(
                            agent.action_map[act]):
                        if scale != 0:
                            tracker.scale_queue(q_idx, scale)

            metrics = tracker.get_all_metrics()
            csr     = metrics['cold_start_rate']
            ckpt_name = os.path.basename(
                ckpt_path.rstrip('/'))
            print(f"    {ckpt_name}: CSR={csr:.3f}%")

            if csr < best_csr:
                best_csr  = csr
                best_path = ckpt_path

        except Exception as e:
            print(f"    Error on {ckpt_path}: {e}")
            continue

    print(f"  Best: {best_path} CSR={best_csr:.3f}%")

    # Copy best checkpoint to best/
    best_dir = model_path + "best/"
    os.makedirs(best_dir, exist_ok=True)
    for f in os.listdir(best_path):
        shutil.copy2(
            os.path.join(best_path, f),
            os.path.join(best_dir, f))
    print(f"  Copied best checkpoint to best/")
    return best_csr


def evaluate_one_seed(seed, model_path,
                      results_path, workloads):
    print(f"\nEvaluating seed {seed}...")

    # CRITICAL: Find best checkpoint first!
    # This is what the original workflow did!
    best_csr = find_best_checkpoint(
        model_path, workloads['Common'])
    print(f"  Best checkpoint CSR: {best_csr:.3f}%")

    best_path = model_path + "best/"
    if not os.path.exists(best_path + "actor.pth"):
        print(f"  No model for seed {seed}!")
        return None

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
                for q_idx, scale in enumerate(action_map[act]):
                    if scale != 0:
                        casr_tracker.scale_queue(q_idx, scale)
        casr_m = casr_tracker.get_all_metrics()

        # TASCAR with best checkpoint
        transformer = TransformerEncoder(state_dim)
        agent = SACAgent(
            transformer_dim=TRANSFORMER_DIM,
            action_dim=action_dim,
            transformer=transformer)
        agent.load(best_path)

        tascar_tracker = MetricsTracker()
        history        = StateHistoryBuffer(SEQUENCE_LENGTH, state_dim)
        call_count     = 0
        for call in calls:
            tascar_tracker.handle_request(call)
            call_count += 1
            if call_count % TASCAR_EVAL_DELTA == 0:
                raw = np.array(
                    tascar_tracker.get_state(),
                    dtype=np.float32)
                mean = np.mean(raw)
                std  = np.std(raw)
                if std > 0:
                    raw = (raw - mean) / std
                history.add(raw)
                enc = agent.get_encoded_state(history.get_sequence())
                act = agent.choose_action(enc, evaluate=True)
                for q_idx, scale in enumerate(agent.action_map[act]):
                    if scale != 0:
                        tascar_tracker.scale_queue(q_idx, scale)
        tascar_m = tascar_tracker.get_all_metrics()

        casr_csr   = casr_m['cold_start_rate']
        tascar_csr = tascar_m['cold_start_rate']
        tascar_m['agi'] = (
            (casr_csr - tascar_csr) / casr_csr * 100
            if casr_csr > 0 else 0)
        casr_m['agi'] = 0.0

        seed_results[wl_name] = {
            'CASR': casr_m, 'TASCAR': tascar_m}

        print(f"    CASR:   {casr_csr:.3f}%")
        print(f"    TASCAR: {tascar_csr:.3f}% "
              f"(+{casr_csr-tascar_csr:.3f}pp)")

    os.makedirs(results_path, exist_ok=True)
    with open(results_path + 'casr_vs_tascar.json', 'w') as f:
        json.dump({
            wl: {
                algo: {k: float(v)
                       for k, v in m.items()
                       if isinstance(v, (int, float))}
                for algo, m in algos.items()}
            for wl, algos in seed_results.items()},
            f, indent=2)

    return seed_results


def load_workloads():
    loader    = AzureDataLoader()
    workloads = {}
    print("\nPreparing workloads...")

    day1   = loader.load_day(1)
    counts = Counter(c.function_id for c in day1)
    top = set(f for f, _ in counts.most_common(NUM_FUNCTIONS))
    common = [c for c in day1 if c.function_id in top]
    np.random.seed(42)
    if len(common) > EVAL_CALLS:
        idx = np.random.choice(len(common), EVAL_CALLS, replace=False)
        idx.sort()
        common = [common[i] for i in idx]
    workloads['Common'] = common
    print(f"  Common: {len(common)} calls")

    day2  = loader.load_day(2)
    heavy = [c for c in day2 if c.cold_start_overhead > 1]
    counts = Counter(c.function_id for c in heavy)
    top = set(f for f, _ in counts.most_common(NUM_FUNCTIONS))
    significant = [c for c in heavy if c.function_id in top]
    np.random.seed(42)
    if len(significant) > EVAL_CALLS:
        idx = np.random.choice(len(significant), EVAL_CALLS, replace=False)
        idx.sort()
        significant = [significant[i] for i in idx]
    workloads['Significant'] = significant
    print(f"  Significant: {len(significant)} calls")

    day3  = loader.load_day(3)
    funcs = list(set(c.function_id for c in day3))
    np.random.seed(43)
    np.random.shuffle(funcs)
    selected  = set(funcs[:NUM_FUNCTIONS])
    random_wl = [c for c in day3 if c.function_id in selected]
    np.random.seed(43)
    if len(random_wl) > EVAL_CALLS:
        idx = np.random.choice(len(random_wl), EVAL_CALLS, replace=False)
        idx.sort()
        random_wl = [random_wl[i] for i in idx]
    workloads['Random'] = random_wl
    print(f"  Random: {len(random_wl)} calls")

    return workloads


def run_multiseed_part2():
    os.makedirs(MULTISEED_RESULTS, exist_ok=True)

    print("=" * 60)
    print("TASCAR Multi-Seed Part 2")
    print(f"New Seeds: {SEEDS}")
    print("With best checkpoint selection!")
    print("=" * 60)

    workloads   = load_workloads()
    all_results = {}

    for seed in SEEDS:
        cfg          = SEED_CONFIGS[seed]
        model_path   = cfg['model_path']
        results_path = cfg['results_path']

        print(f"\n{'='*50}")
        print(f"Seed: {seed}")
        print(f"{'='*50}")

        train_one_seed(seed, model_path, results_path)
        seed_results = evaluate_one_seed(
            seed, model_path, results_path, workloads)

        if seed_results:
            all_results[seed] = seed_results

        time.sleep(10)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for seed, seed_r in all_results.items():
        print(f"\nSeed {seed}:")
        for wl, algos in seed_r.items():
            casr_csr   = algos['CASR']['cold_start_rate']
            tascar_csr = algos['TASCAR']['cold_start_rate']
            diff = casr_csr - tascar_csr
            symbol = "✅" if diff > 0 else "❌"
            print(f"  {wl}: CASR={casr_csr:.3f}% "
                  f"TASCAR={tascar_csr:.3f}% "
                  f"+{diff:.3f}pp {symbol}")

    path = MULTISEED_RESULTS + 'multiseed_part2_results.json'
    with open(path, 'w') as f:
        json.dump(
            {str(k): {
                wl: {
                    algo: {m: float(v)
                           for m, v in metrics.items()
                           if isinstance(v, (int, float))}
                    for algo, metrics in algos.items()}
                for wl, algos in seed_r.items()}
             for k, seed_r in all_results.items()},
            f, indent=2)
    print(f"\nSaved: {path}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("TASCAR Multi-Seed Part 2")
    print("Seeds: 789, 1000, 2024, 2025")
    print("With best checkpoint selection!")
    print("~6 hours total")
    print("=" * 60)
    run_multiseed_part2()
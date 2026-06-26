# run_gru_sac_seed456_fixed.py
# Trains GRU+SAC seed 123
# Saves to trained_model_gru_sac_seed456/

import config
config.RANDOM_SEED    = 456
config.TASCAR_MODEL_PATH = "trained_model_gru_sac_seed456/"
config.TASCAR_RESULTS    = "results_gru_sac_seed456/"

import numpy as np
import os
import torch
from collections import Counter

from simulator import AzureDataLoader
from scache import SCache
from gru_encoder import GRUEncoder
from transformer_encoder import StateHistoryBuffer
from sac_agent import SACAgent
from train_tascar import (
    load_filtered_data,
    normalize_state,
    compute_dynamic_theta,
    RewardNormalizer,
    TASCARLogger
)
from config import (
    NUM_QUEUES, EVAL_CALLS, TASCAR_DELTA,
    THETA, SEQUENCE_LENGTH, TRANSFORMER_DIM,
    TASCAR_EPISODES, SAC_UPDATES_PER_STEP,
    RANDOM_SEED
)

MODEL_PATH   = "trained_model_gru_sac_seed456/"
RESULTS_PATH = "results_gru_sac_seed456/"

print("=" * 55)
print("GRU+SAC Training (V6 Ablation)")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Episodes: {TASCAR_EPISODES}")
print(f"Save path: {MODEL_PATH}")
print("=" * 55)

os.makedirs(MODEL_PATH,   exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print(f"Random seed: {RANDOM_SEED}")
print("Encoder: GRU (V6 ablation)")

print("\nLoading Azure dataset...")
train_data = load_filtered_data()

state_dim  = NUM_QUEUES * 7
action_dim = 3 ** NUM_QUEUES

gru  = GRUEncoder(state_dim)
agent = SACAgent(
    transformer_dim=TRANSFORMER_DIM,
    action_dim=action_dim,
    transformer=gru)

data_rng = np.random.RandomState(42)

print("\nWarming up buffer...")
for ep in range(20):
    max_start = max(1, len(train_data) - EVAL_CALLS)
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
    seq = history.get_sequence()
    enc = gru(torch.FloatTensor(seq).unsqueeze(0))

    call_count = wmt_before = 0
    step_cold  = step_warm  = 0

    for call in episode_calls:
        is_warm = scache.handle_request(call)
        if is_warm: step_warm += 1
        else:       step_cold += 1
        call_count += 1

        if call_count % TASCAR_DELTA == 0:
            new_raw = normalize_state(
                scache.get_state())
            history.add(new_raw)
            next_enc = gru(
                torch.FloatTensor(
                    history.get_sequence()
                ).unsqueeze(0))

            action = np.random.randint(
                0, action_dim)
            total  = step_cold + step_warm
            cr     = (step_cold / total
                      if total > 0 else 0)
            cwmt   = (scache
                      .get_total_wasted_memory_time())
            wmt_change = max(0, cwmt - wmt_before)
            wmt_before = cwmt
            reward = -(
                THETA * min(cr, 1.0) +
                (1 - THETA) *
                min(wmt_change / 100.0, 1.0))

            enc_np  = enc.detach().numpy()
            next_np = next_enc.detach().numpy()
            if (not np.isnan(enc_np).any() and
                    not np.isnan(next_np).any()):
                agent.store_experience(
                    enc_np, action,
                    reward, next_np, False)

            enc = next_enc
            step_cold = step_warm = 0

    print(f"  Warmup ep {ep+1:2d}: "
          f"Buffer: {len(agent.buffer)}")

print(f"Warmup complete! Buffer: {len(agent.buffer)}")

reward_norm   = RewardNormalizer()
logger        = TASCARLogger()
current_theta = THETA
logger.start_training()

print(f"\nStarting training...")
print("=" * 55)

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
    encoded_state = gru(
        torch.FloatTensor(
            history.get_sequence()
        ).unsqueeze(0)).detach().numpy()

    ep_reward = ep_cold = ep_warm = 0.0
    step_cold = step_warm = 0
    call_count = steps_done = 0
    wmt_before = 0.0
    ep_actor_loss  = []
    ep_critic_loss = []

    for call in episode_calls:
        is_warm = scache.handle_request(call)
        if is_warm: step_warm += 1; ep_warm += 1
        else:       step_cold += 1; ep_cold += 1
        call_count += 1

        if call_count % TASCAR_DELTA == 0:
            new_raw = normalize_state(
                scache.get_state())
            history.add(new_raw)
            next_encoded = gru(
                torch.FloatTensor(
                    history.get_sequence()
                ).unsqueeze(0)).detach().numpy()

            total = step_cold + step_warm
            cr    = (step_cold / total
                     if total > 0 else 0)
            current_theta = compute_dynamic_theta(
                cr, current_theta)

            cwmt = (scache
                    .get_total_wasted_memory_time())
            wmt_change = max(0, cwmt - wmt_before)
            wmt_before = cwmt

            reward = reward_norm.calculate(
                step_cold, wmt_change,
                current_theta)
            action = agent.choose_action(
                encoded_state)

            if (not np.isnan(encoded_state).any()
                    and not np.isnan(
                        next_encoded).any()):
                agent.store_experience(
                    encoded_state, action,
                    reward, next_encoded, False)

            ep_reward  += reward
            steps_done += 1

            for _ in range(SAC_UPDATES_PER_STEP):
                result = agent.update()
                if result[0] is not None:
                    ep_actor_loss.append(result[0])
                    ep_critic_loss.append(result[1])

            scales = agent.action_map[action]
            for q_idx, scale in enumerate(scales):
                if scale != 0:
                    scache.scale_queue(
                        q_idx, scale)

            encoded_state = next_encoded
            step_cold = step_warm = 0

    total_calls = ep_cold + ep_warm
    cold_pct    = (ep_cold / total_calls * 100
                   if total_calls > 0 else 0)
    avg_reward  = (ep_reward / steps_done
                   if steps_done > 0 else ep_reward)
    avg_actor   = (np.mean(ep_actor_loss)
                   if ep_actor_loss else 0)
    avg_critic  = (np.mean(ep_critic_loss)
                   if ep_critic_loss else 0)

    logger.log_episode(
        episode, avg_reward, cold_pct,
        scache.get_total_wasted_memory_time(),
        current_theta, avg_actor, avg_critic,
        steps_this_ep=steps_done)

    if avg_reward > logger.best_reward:
        agent.save(MODEL_PATH + "best/")

    if episode % 10 == 0:
        avg_r = np.mean(logger.rewards[-10:])
        avg_c = np.mean(
            logger.cold_start_rates[-10:])
        elapsed = logger.get_training_time()
        print(f"Ep {episode:3d} | "
              f"Reward: {avg_r:7.4f} | "
              f"Cold%: {avg_c:5.1f}% | "
              f"Theta: {current_theta:.3f} | "
              f"Buffer: {len(agent.buffer):5d} | "
              f"Time: {elapsed:.0f}s")

    if episode % 50 == 0:
        agent.save(
            MODEL_PATH +
            f"checkpoint_ep{episode}/")
        logger.save_logs(RESULTS_PATH)
        logger.plot_training(RESULTS_PATH)

logger.end_training()
agent.save(MODEL_PATH + "best/")
logger.save_logs(RESULTS_PATH)
logger.plot_training(RESULTS_PATH)

rl = logger.get_rl_metrics()
print("\n" + "=" * 55)
print("GRU+SAC Training Complete!")
print("=" * 55)
print(f"Random seed:       {RANDOM_SEED}")
print(f"Best reward:       {rl['best_reward']:.4f}")
print(f"Training time:     {rl['training_time_seconds']:.1f}s")
print(f"Convergence ep:    {rl['convergence_episode']}")
print("=" * 55)
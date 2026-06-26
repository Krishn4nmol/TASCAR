# train_lstm_sac.py
# Trains LSTM+SAC variant (V5)
# Drop-in replacement of Transformer with LSTM
# Used for ablation: is Transformer better than LSTM?

import config
config.TASCAR_MODEL_PATH = "trained_model_lstm_sac/"
config.TASCAR_RESULTS    = "results_lstm_sac/"

import numpy as np
import os
import time
from collections import Counter

from simulator import AzureDataLoader
from scache import SCache
from lstm_encoder import LSTMEncoder
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
    TASCAR_EPISODES, TASCAR_MODEL_PATH,
    TASCAR_RESULTS, SAC_BATCH_SIZE,
    SAC_UPDATES_PER_STEP, RANDOM_SEED
)


def train_lstm_sac():
    os.makedirs(TASCAR_MODEL_PATH, exist_ok=True)
    os.makedirs(TASCAR_RESULTS, exist_ok=True)

    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")
    print("Encoder: LSTM (V5 ablation)")

    print("\nLoading Azure dataset...")
    train_data = load_filtered_data()

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    print(f"\nLSTM+SAC Configuration:")
    print(f"  State dim:    {state_dim}")
    print(f"  Action dim:   {action_dim}")
    print(f"  Encoder:      LSTM (bidirectional, 2 layers)")
    print(f"  Episodes:     {TASCAR_EPISODES}")
    print(f"  Random seed:  {RANDOM_SEED}")

    # Use LSTM encoder instead of Transformer
    lstm = LSTMEncoder(state_dim)
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=lstm)   # SAC agent accepts any encoder

    data_rng = np.random.RandomState(42)

    # Warmup buffer
    print("\nWarming up buffer...")
    for ep in range(20):
        max_start = max(1, len(train_data) - EVAL_CALLS)
        start_idx = data_rng.randint(0, max_start)
        episode_calls = train_data[start_idx:start_idx + EVAL_CALLS]
        if len(episode_calls) < 1000:
            episode_calls = train_data[:EVAL_CALLS]

        scache  = SCache()
        history = StateHistoryBuffer(SEQUENCE_LENGTH, state_dim)
        raw     = normalize_state(scache.get_state())
        history.add(raw)
        seq     = history.get_sequence()
        seq_t   = lstm(
            __import__('torch').FloatTensor(seq).unsqueeze(0))

        call_count = 0
        wmt_before = 0.0
        step_cold  = 0
        step_warm  = 0

        for call in episode_calls:
            is_warm = scache.handle_request(call)
            if is_warm: step_warm += 1
            else:       step_cold += 1
            call_count += 1

            if call_count % TASCAR_DELTA == 0:
                new_raw  = normalize_state(scache.get_state())
                history.add(new_raw)
                new_seq  = history.get_sequence()
                import torch
                next_enc = lstm(
                    torch.FloatTensor(new_seq).unsqueeze(0))

                action = np.random.randint(0, action_dim)
                total  = step_cold + step_warm
                cr     = step_cold / total if total > 0 else 0
                cwmt   = scache.get_total_wasted_memory_time()
                wmt_change = max(0, cwmt - wmt_before)
                wmt_before = cwmt

                reward = -(THETA * min(cr, 1.0) +
                           (1 - THETA) * min(wmt_change / 100.0, 1.0))

                enc_np  = seq_t.detach().numpy()
                next_np = next_enc.detach().numpy()
                if not (np.isnan(enc_np).any() or np.isnan(next_np).any()):
                    agent.store_experience(enc_np, action, reward, next_np, False)

                seq_t = next_enc
                step_cold = step_warm = 0

        print(f"  Warmup ep {ep+1:2d}: Buffer: {len(agent.buffer)}")

    print(f"Warmup complete! Buffer: {len(agent.buffer)}")

    # Training loop
    reward_norm   = RewardNormalizer()
    logger        = TASCARLogger()
    calls_per_ep  = EVAL_CALLS
    current_theta = THETA
    logger.start_training()

    print(f"\nStarting training...")
    print("=" * 55)

    import torch

    for episode in range(1, TASCAR_EPISODES + 1):
        max_start    = max(1, len(train_data) - calls_per_ep)
        start_idx    = data_rng.randint(0, max_start)
        episode_calls = train_data[start_idx:start_idx + calls_per_ep]
        if len(episode_calls) < 1000:
            episode_calls = train_data[:calls_per_ep]

        scache  = SCache()
        history = StateHistoryBuffer(SEQUENCE_LENGTH, state_dim)
        raw     = normalize_state(scache.get_state())
        history.add(raw)
        seq     = history.get_sequence()
        encoded = lstm(torch.FloatTensor(seq).unsqueeze(0))
        encoded_state = encoded.detach().numpy()

        ep_reward = ep_cold = ep_warm = 0.0
        step_cold = step_warm = call_count = steps_done = 0
        wmt_before = 0.0
        ep_actor_loss = []
        ep_critic_loss = []

        for call in episode_calls:
            is_warm = scache.handle_request(call)
            if is_warm: step_warm += 1; ep_warm += 1
            else:       step_cold += 1; ep_cold += 1
            call_count += 1

            if call_count % TASCAR_DELTA == 0:
                new_raw = normalize_state(scache.get_state())
                history.add(new_raw)
                new_seq  = history.get_sequence()
                next_enc = lstm(
                    torch.FloatTensor(new_seq).unsqueeze(0))
                next_encoded = next_enc.detach().numpy()

                total = step_cold + step_warm
                cr    = step_cold / total if total > 0 else 0
                current_theta = compute_dynamic_theta(cr, current_theta)

                cwmt       = scache.get_total_wasted_memory_time()
                wmt_change = max(0, cwmt - wmt_before)
                wmt_before = cwmt

                reward = reward_norm.calculate(
                    step_cold, wmt_change, current_theta)
                action = agent.choose_action(encoded_state)

                if (not np.isnan(encoded_state).any() and
                        not np.isnan(next_encoded).any()):
                    agent.store_experience(
                        encoded_state, action, reward,
                        next_encoded, False)

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
                        scache.scale_queue(q_idx, scale)

                encoded_state = next_encoded
                step_cold = step_warm = 0

        total_calls = ep_cold + ep_warm
        cold_pct    = ep_cold / total_calls * 100 if total_calls > 0 else 0
        avg_reward  = ep_reward / steps_done if steps_done > 0 else ep_reward
        avg_actor   = np.mean(ep_actor_loss) if ep_actor_loss else 0
        avg_critic  = np.mean(ep_critic_loss) if ep_critic_loss else 0

        logger.log_episode(
            episode, avg_reward, cold_pct,
            scache.get_total_wasted_memory_time(),
            current_theta, avg_actor, avg_critic,
            steps_this_ep=steps_done)

        if avg_reward > logger.best_reward:
            agent.save(TASCAR_MODEL_PATH + "best/")

        if episode % 10 == 0:
            avg_r = np.mean(logger.rewards[-10:])
            avg_c = np.mean(logger.cold_start_rates[-10:])
            elapsed = logger.get_training_time()
            print(f"Ep {episode:3d} | Reward: {avg_r:7.4f} | "
                  f"Cold%: {avg_c:5.1f}% | Theta: {current_theta:.3f} | "
                  f"Buffer: {len(agent.buffer):5d} | Time: {elapsed:.0f}s")

        if episode % 50 == 0:
            agent.save(TASCAR_MODEL_PATH + f"checkpoint_ep{episode}/")
            logger.save_logs(TASCAR_RESULTS)
            logger.plot_training(TASCAR_RESULTS)

    logger.end_training()
    agent.save(TASCAR_MODEL_PATH + "best/")
    logger.save_logs(TASCAR_RESULTS)
    logger.plot_training(TASCAR_RESULTS)

    rl = logger.get_rl_metrics()
    print("\n" + "=" * 55)
    print("LSTM+SAC Training Complete!")
    print("=" * 55)
    print(f"Random seed:       {RANDOM_SEED}")
    print(f"Best reward:       {rl['best_reward']:.4f}")
    print(f"Training time:     {rl['training_time_seconds']:.1f}s")
    print(f"Convergence ep:    {rl['convergence_episode']}")
    print("=" * 55)

    return agent, logger


if __name__ == "__main__":
    print("=" * 55)
    print("LSTM+SAC Training (V5 Ablation)")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Episodes: {TASCAR_EPISODES}")
    print("=" * 55)
    train_lstm_sac()
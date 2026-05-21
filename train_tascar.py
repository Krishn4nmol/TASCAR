# train_tascar.py
# Training script for TASCAR
# Uses SAC + Transformer encoder
# Fixed with proper delta and warmup!

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from config import (
    NUM_QUEUES,
    TRAIN_DAYS,
    NUM_FUNCTIONS,
    EVAL_CALLS,
    DELTA,
    TASCAR_DELTA,
    THETA,
    THETA_MIN,
    THETA_MAX,
    THETA_ADAPT_RATE,
    SEQUENCE_LENGTH,
    TRANSFORMER_DIM,
    TASCAR_EPISODES,
    TASCAR_MODEL_PATH,
    TASCAR_RESULTS,
    SAC_BATCH_SIZE,
    SAC_UPDATES_PER_STEP
)
from simulator import AzureDataLoader
from scache import SCache
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

def load_filtered_data():
    """
    Loads Azure dataset filtered
    to top NUM_FUNCTIONS functions.
    Same as CASR train.py!
    """
    loader     = AzureDataLoader()
    train_data = []

    for day in TRAIN_DAYS:
        print(f"  Loading day {day}...")
        day_calls = loader.load_day(day)
        train_data.extend(day_calls)

    print(f"  Total before filter: "
          f"{len(train_data)}")

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

    print(f"  Total after filter: "
          f"{len(train_data)}")
    print(f"  Unique functions: "
          f"{NUM_FUNCTIONS}")

    return train_data


# ─────────────────────────────────────────
# NORMALIZE STATE
# ─────────────────────────────────────────

def normalize_state(raw_state):
    """
    Normalize state to zero mean
    unit variance.
    Same as CASR environment.py!
    """
    state = np.array(
        raw_state,
        dtype=np.float32)
    mean = np.mean(state)
    std  = np.std(state)
    if std > 0:
        state = (state - mean) / std
    return state


# ─────────────────────────────────────────
# DYNAMIC THETA
# Key innovation of TASCAR!
# ─────────────────────────────────────────

def compute_dynamic_theta(
        cold_start_rate,
        current_theta):
    """
    Adapts theta based on performance.

    Cold starts too high (>90%):
    Increase theta → focus on cold starts!

    Cold starts acceptable (<70%):
    Decrease theta → focus on memory!

    CASR uses fixed theta=0.8 always.
    TASCAR adapts automatically!
    """
    if cold_start_rate > 0.9:
        new_theta = min(
            current_theta +
            THETA_ADAPT_RATE,
            THETA_MAX)
    elif cold_start_rate < 0.7:
        new_theta = max(
            current_theta -
            THETA_ADAPT_RATE,
            THETA_MIN)
    else:
        new_theta = current_theta
    return new_theta


# ─────────────────────────────────────────
# REWARD NORMALIZER
# Same normalization as CASR!
# Fixes reward scale mismatch!
# ─────────────────────────────────────────

class RewardNormalizer:
    """
    Normalizes reward exactly like
    CASR environment.py does!

    Without this TASCAR reward was -8.3
    CASR reward was -0.04
    Not comparable!

    With this both rewards are
    in same range -1.0 to 0.0
    Fair comparison!
    """
    def __init__(self):
        self.r1_min =  float('inf')
        self.r1_max = -float('inf')
        self.r2_min =  float('inf')
        self.r2_max = -float('inf')

    def calculate(self,
                  cold_starts,
                  wmt_change,
                  theta):
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

        r1_range = self.r1_max - self.r1_min
        r1_norm  = (
            (r1 - self.r1_min) /
            r1_range
            if r1_range > 0
            else 0.0)

        r2_range = self.r2_max - self.r2_min
        r2_norm  = (
            (r2 - self.r2_min) /
            r2_range
            if r2_range > 0
            else 0.0)

        reward = -(
            theta * r1_norm +
            (1 - theta) * r2_norm)

        return float(reward)


# ─────────────────────────────────────────
# TRAINING LOGGER
# ─────────────────────────────────────────

class TASCARLogger:
    """
    Records training metrics.
    Same key names as CASR for
    consistency!
    """
    def __init__(self):
        self.episodes         = []
        self.rewards          = []
        self.cold_start_rates = []
        self.wmts             = []
        self.thetas           = []
        self.actor_losses     = []
        self.critic_losses    = []
        self.best_reward      = float('-inf')

    def log_episode(self,
                    episode,
                    reward,
                    cold_rate,
                    wmt,
                    theta,
                    actor_loss=None,
                    critic_loss=None):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.cold_start_rates.append(
            cold_rate)
        self.wmts.append(wmt)
        self.thetas.append(theta)
        if actor_loss is not None:
            self.actor_losses.append(
                actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(
                critic_loss)
        if reward > self.best_reward:
            self.best_reward = reward

    def save_logs(self, path):
        os.makedirs(
            path, exist_ok=True)
        logs = {
            'episodes':         self.episodes,
            'rewards':          self.rewards,
            'cold_start_rates': self.cold_start_rates,
            'wmts':             self.wmts,
            'thetas':           self.thetas,
            'actor_losses':     self.actor_losses,
            'critic_losses':    self.critic_losses,
            'best_reward':      self.best_reward
        }
        with open(
                path +
                'training_logs.json',
                'w') as f:
            json.dump(
                logs, f, indent=2)
        print(f"Logs saved!")

    def plot_training(self, path):
        if len(self.episodes) < 2:
            return

        fig, axes = plt.subplots(
            2, 2, figsize=(14, 10))
        fig.suptitle(
            'TASCAR Training Progress',
            fontsize=14,
            fontweight='bold')

        axes[0, 0].plot(
            self.episodes,
            self.rewards,
            color='blue',
            alpha=0.4,
            linewidth=1)
        axes[0, 0].plot(
            self.episodes,
            self._smooth(
                self.rewards, 10),
            color='darkblue',
            linewidth=2.5,
            label='Smoothed')
        axes[0, 0].set_title(
            'Reward Convergence')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(
            self.episodes,
            self.cold_start_rates,
            color='red',
            alpha=0.4,
            linewidth=1)
        axes[0, 1].plot(
            self.episodes,
            self._smooth(
                self.cold_start_rates,
                10),
            color='darkred',
            linewidth=2.5,
            label='Smoothed')
        axes[0, 1].set_title(
            'Cold Start Rate (%)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cold%')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(
            self.episodes,
            self.wmts,
            color='green',
            alpha=0.4,
            linewidth=1)
        axes[1, 0].plot(
            self.episodes,
            self._smooth(
                self.wmts, 10),
            color='darkgreen',
            linewidth=2.5,
            label='Smoothed')
        axes[1, 0].set_title(
            'Wasted Memory Time (s)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('WMT (s)')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(
            self.episodes,
            self.thetas,
            color='purple',
            linewidth=2,
            label='TASCAR theta')
        axes[1, 1].axhline(
            y=0.8,
            color='red',
            linestyle='--',
            label='CASR fixed=0.8')
        axes[1, 1].set_title(
            'Dynamic Theta')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Theta')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_ylim(0.4, 1.0)

        plt.tight_layout()
        plt.savefig(
            path +
            'tascar_training.png',
            dpi=150,
            bbox_inches='tight')
        plt.close()
        print("Training graph saved!")

    def _smooth(self, values, window):
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window)
            smoothed.append(
                np.mean(
                    values[start:i+1]))
        return smoothed


# ─────────────────────────────────────────
# WARMUP BUFFER
# Fill buffer before training!
# Critical for SAC learning!
# ─────────────────────────────────────────

def warmup_buffer(agent,
                  train_data,
                  state_dim,
                  warmup_episodes=20):
    """
    Fills replay buffer with random
    experiences before training starts!

    Why needed:
    SAC is OFF-POLICY
    Needs diverse experiences first
    Before policy gradient updates work!

    CASR PPO is ON-POLICY
    Does not need this warmup!

    With TASCAR_DELTA=1000:
    Each episode = 100 steps
    20 warmup episodes = 2000 experiences
    Enough to start learning!
    """
    steps_per_ep = (
        EVAL_CALLS // TASCAR_DELTA)

    print(
        f"\nWarming up buffer...")
    print(
        f"  Episodes:        "
        f"{warmup_episodes}")
    print(
        f"  Steps/episode:   "
        f"{steps_per_ep}")
    print(
        f"  Expected buffer: "
        f"{warmup_episodes * steps_per_ep}")

    for ep in range(warmup_episodes):
        max_start = max(
            1,
            len(train_data) - EVAL_CALLS)
        start_idx = np.random.randint(
            0, max_start)
        episode_calls = train_data[
            start_idx:
            start_idx + EVAL_CALLS]

        if len(episode_calls) < 1000:
            episode_calls = (
                train_data[:EVAL_CALLS])

        scache  = SCache()
        history = StateHistoryBuffer(
            SEQUENCE_LENGTH, state_dim)

        raw     = normalize_state(
            scache.get_state())
        history.add(raw)
        seq     = history.get_sequence()
        encoded = (
            agent.get_encoded_state(seq))

        step_cold  = 0
        step_warm  = 0
        call_count = 0
        wmt_before = 0.0

        for call in episode_calls:
            is_warm = (
                scache.handle_request(
                    call))
            if is_warm:
                step_warm += 1
            else:
                step_cold += 1
            call_count += 1

            if (call_count %
                    TASCAR_DELTA == 0):

                new_raw = normalize_state(
                    scache.get_state())
                history.add(new_raw)
                new_seq = (
                    history.get_sequence())
                next_enc = (
                    agent.get_encoded_state(
                        new_seq))

                # Random action!
                action = np.random.randint(
                    0, agent.action_dim)

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

                reward = -(
                    THETA *
                    min(cold_rate, 1.0) +
                    (1 - THETA) *
                    min(
                        wmt_change /
                        100.0, 1.0))

                agent.store_experience(
                    encoded,
                    action,
                    reward,
                    next_enc,
                    False)

                # Apply random action
                scales = (
                    agent.action_map[
                        action])
                for q_idx, scale in (
                        enumerate(scales)):
                    if scale != 0:
                        scache.scale_queue(
                            q_idx, scale)

                encoded   = next_enc
                step_cold = 0
                step_warm = 0

        print(
            f"  Warmup ep {ep+1:2d}: "
            f"Buffer: "
            f"{len(agent.buffer)}")

    print(
        f"Warmup complete! "
        f"Buffer: {len(agent.buffer)}")


# ─────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────

def train_tascar():
    """
    Main TASCAR training loop.

    Key fixes over previous version:
    1. TASCAR_DELTA=1000 (was 10000)
       100 steps/episode (was 10)
    2. Buffer warmup phase
       2000 experiences before learning
    3. Proper reward normalization
       Same scale as CASR!
    4. 10 SAC updates per step
       Faster learning!
    """
    os.makedirs(
        TASCAR_MODEL_PATH,
        exist_ok=True)
    os.makedirs(
        TASCAR_RESULTS,
        exist_ok=True)

    print("\nLoading Azure dataset...")
    train_data = load_filtered_data()

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    print(f"\nTASCAR Configuration:")
    print(f"  State dim:       {state_dim}")
    print(f"  Action dim:      {action_dim}")
    print(f"  TASCAR delta:    {TASCAR_DELTA}")
    print(f"  Steps/episode:   "
          f"{EVAL_CALLS // TASCAR_DELTA}")
    print(f"  Sequence length: "
          f"{SEQUENCE_LENGTH}")
    print(f"  Transformer dim: "
          f"{TRANSFORMER_DIM}")
    print(f"  Episodes:        "
          f"{TASCAR_EPISODES}")
    print(f"  Updates/step:    "
          f"{SAC_UPDATES_PER_STEP}")
    print(f"  Batch size:      "
          f"{SAC_BATCH_SIZE}")

    # Create Transformer
    transformer = TransformerEncoder(
        state_dim)

    # Create SAC agent
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=transformer)

    # Warmup buffer!
    warmup_buffer(
        agent,
        train_data,
        state_dim,
        warmup_episodes=20)

    # Reward normalizer
    reward_norm   = RewardNormalizer()
    logger        = TASCARLogger()
    calls_per_ep  = EVAL_CALLS
    current_theta = THETA

    print(f"\nStarting training...")
    print("=" * 55)

    for episode in range(
            1, TASCAR_EPISODES + 1):

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

        scache  = SCache()
        history = StateHistoryBuffer(
            SEQUENCE_LENGTH, state_dim)

        raw_state = normalize_state(
            scache.get_state())
        history.add(raw_state)
        seq = history.get_sequence()
        encoded_state = (
            agent.get_encoded_state(seq))

        ep_reward      = 0.0
        ep_cold        = 0
        ep_warm        = 0
        step_cold      = 0
        step_warm      = 0
        call_count     = 0
        wmt_before     = 0.0
        ep_actor_loss  = []
        ep_critic_loss = []

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

            # TASCAR_DELTA = 1000
            # 100 decisions per episode!
            if (call_count %
                    TASCAR_DELTA == 0):

                new_raw = normalize_state(
                    scache.get_state())
                history.add(new_raw)
                new_seq = (
                    history.get_sequence())
                next_encoded = (
                    agent.get_encoded_state(
                        new_seq))

                # Dynamic theta
                total = (
                    step_cold + step_warm)
                cold_rate = (
                    step_cold / total
                    if total > 0
                    else 0)
                current_theta = (
                    compute_dynamic_theta(
                        cold_rate,
                        current_theta))

                # Normalized reward!
                current_wmt = (
                    scache
                    .get_total_wasted_memory_time())
                wmt_change = max(
                    0,
                    current_wmt -
                    wmt_before)
                wmt_before = current_wmt

                reward = (
                    reward_norm.calculate(
                        step_cold,
                        wmt_change,
                        current_theta))

                # Pick action
                action = (
                    agent.choose_action(
                        encoded_state))

                # Store experience
                agent.store_experience(
                    encoded_state,
                    action,
                    reward,
                    next_encoded,
                    False)

                ep_reward += reward

                # Multiple updates!
                for _ in range(
                        SAC_UPDATES_PER_STEP):
                    result = agent.update()
                    if result[0] is not None:
                        ep_actor_loss.append(
                            result[0])
                        ep_critic_loss.append(
                            result[1])

                # Apply action
                scales = (
                    agent.action_map[
                        action])
                for q_idx, scale in (
                        enumerate(scales)):
                    if scale != 0:
                        scache.scale_queue(
                            q_idx, scale)

                encoded_state = (
                    next_encoded)
                step_cold = 0
                step_warm = 0

        # Episode metrics
        total_calls = ep_cold + ep_warm
        cold_pct = (
            ep_cold /
            total_calls * 100
            if total_calls > 0
            else 0)
        final_wmt = (
            scache
            .get_total_wasted_memory_time())
        avg_actor = (
            np.mean(ep_actor_loss)
            if ep_actor_loss
            else 0)
        avg_critic = (
            np.mean(ep_critic_loss)
            if ep_critic_loss
            else 0)

        logger.log_episode(
            episode,
            ep_reward,
            cold_pct,
            final_wmt,
            current_theta,
            avg_actor,
            avg_critic)

        if ep_reward > logger.best_reward:
            agent.save(
                TASCAR_MODEL_PATH +
                "best/")

        if episode % 10 == 0:
            avg_r = np.mean(
                logger.rewards[-10:])
            avg_c = np.mean(
                logger.cold_start_rates
                [-10:])
            print(
                f"Ep {episode:3d} | "
                f"Reward: {avg_r:7.4f} | "
                f"Cold%: {avg_c:5.1f}% | "
                f"Theta: "
                f"{current_theta:.3f} | "
                f"Buffer: "
                f"{len(agent.buffer):5d}")

        if episode % 50 == 0:
            agent.save(
                TASCAR_MODEL_PATH +
                f"checkpoint_ep"
                f"{episode}/")
            logger.save_logs(
                TASCAR_RESULTS)
            logger.plot_training(
                TASCAR_RESULTS)

    # Final save
    agent.save(
        TASCAR_MODEL_PATH + "best/")
    logger.save_logs(TASCAR_RESULTS)
    logger.plot_training(TASCAR_RESULTS)

    print("\n" + "=" * 55)
    print("TASCAR Training Complete!")
    print(f"Best reward: "
          f"{logger.best_reward:.4f}")
    print(f"Final theta: "
          f"{current_theta:.3f}")
    print(f"Steps/episode: "
          f"{EVAL_CALLS // TASCAR_DELTA}")
    print(f"Total steps: "
          f"{TASCAR_EPISODES * (EVAL_CALLS // TASCAR_DELTA)}")
    print("=" * 55)

    return agent, logger


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("TASCAR Training")
    print("Transformer-Attention SAC")
    print("for Serverless Computing")
    print("=" * 55)
    train_tascar()
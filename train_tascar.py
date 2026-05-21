# train_tascar.py
# Training script for TASCAR
# Uses SAC + Transformer encoder
# Replaces PPO training from CASR

import numpy as np
import json
import os
import sys
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
    THETA,
    THETA_MIN,
    THETA_MAX,
    THETA_ADAPT_RATE,
    SEQUENCE_LENGTH,
    TRANSFORMER_DIM,
    TASCAR_EPISODES,
    TASCAR_MODEL_PATH,
    TASCAR_RESULTS,
    SAC_BATCH_SIZE
)
from simulator import AzureDataLoader
from scache import SCache
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent


# ─────────────────────────────────────────
# LOAD DATA
# Same as CASR train.py
# Filters to top 2000 functions
# ─────────────────────────────────────────

def load_filtered_data():
    """
    Loads Azure dataset and filters
    to top NUM_FUNCTIONS functions.
    Same approach as CASR train.py!
    """
    loader     = AzureDataLoader()
    train_data = []

    for day in TRAIN_DAYS:
        print(f"  Loading day {day}...")
        day_calls = loader.load_day(day)
        train_data.extend(day_calls)

    print(f"  Total calls before filter: "
          f"{len(train_data)}")

    # Filter to top functions
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

    print(f"  Total calls after filter: "
          f"{len(train_data)}")
    print(f"  Unique functions: "
          f"{NUM_FUNCTIONS}")

    return train_data


# ─────────────────────────────────────────
# DYNAMIC THETA
# Key innovation of TASCAR!
# Theta adapts based on performance!
# CASR has fixed theta = 0.8
# TASCAR adjusts automatically!
# ─────────────────────────────────────────

def compute_dynamic_theta(
        cold_start_rate,
        current_theta):
    """
    Adapts theta based on cold start rate.

    If cold starts too high (>90%):
    Increase theta → focus more on
    reducing cold starts!

    If cold starts acceptable (<70%):
    Decrease theta → focus more on
    memory efficiency!

    This is key improvement over CASR
    which uses fixed theta = 0.8 always!
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
# TRAINING LOGGER
# Records training progress
# Generates graphs after training
# ─────────────────────────────────────────

class TASCARLogger:
    """
    Records all training metrics
    for analysis and graphing.
    """
    def __init__(self):
        self.episodes     = []
        self.rewards      = []
        self.cold_rates   = []
        self.wmts         = []
        self.thetas       = []
        self.actor_losses = []
        self.critic_losses = []
        self.best_reward  = float('-inf')

    def log_episode(self, episode,
                    reward, cold_rate,
                    wmt, theta,
                    actor_loss=None,
                    critic_loss=None):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.cold_rates.append(cold_rate)
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
        """Save logs to JSON file"""
        os.makedirs(path, exist_ok=True)
        logs = {
            'episodes':      self.episodes,
            'rewards':       self.rewards,
            'cold_rates':    self.cold_rates,
            'wmts':          self.wmts,
            'thetas':        self.thetas,
            'actor_losses':  self.actor_losses,
            'critic_losses': self.critic_losses,
            'best_reward':   self.best_reward
        }
        with open(
                path + 'training_logs.json',
                'w') as f:
            json.dump(logs, f, indent=2)
        print(f"Logs saved to {path}")

    def plot_training(self, path):
        """Generate training graphs"""
        fig, axes = plt.subplots(
            2, 2, figsize=(14, 10))
        fig.suptitle(
            'TASCAR Training Progress',
            fontsize=14,
            fontweight='bold')

        # Reward convergence
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

        # Cold start rate
        axes[0, 1].plot(
            self.episodes,
            self.cold_rates,
            color='red',
            alpha=0.4,
            linewidth=1)
        axes[0, 1].plot(
            self.episodes,
            self._smooth(
                self.cold_rates, 10),
            color='darkred',
            linewidth=2.5,
            label='Smoothed')
        axes[0, 1].set_title(
            'Cold Start Rate (%)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cold%')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # WMT
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

        # Dynamic theta
        axes[1, 1].plot(
            self.episodes,
            self.thetas,
            color='purple',
            linewidth=2)
        axes[1, 1].axhline(
            y=0.8,
            color='red',
            linestyle='--',
            label='CASR fixed theta=0.8')
        axes[1, 1].set_title(
            'Dynamic Theta Value')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Theta')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_ylim(0.4, 1.0)

        plt.tight_layout()
        plt.savefig(
            path + 'tascar_training.png',
            dpi=150,
            bbox_inches='tight')
        plt.close()
        print("Training graph saved!")

    def _smooth(self, values, window):
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window)
            smoothed.append(
                np.mean(values[start:i+1]))
        return smoothed


# ─────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────

def train_tascar():
    """
    Main TASCAR training loop.

    Key differences from CASR train.py:
    1. Uses SAC instead of PPO
    2. Transformer encodes state sequence
    3. Dynamic theta adaptation
    4. Off-policy replay buffer
    """
    os.makedirs(
        TASCAR_MODEL_PATH,
        exist_ok=True)
    os.makedirs(
        TASCAR_RESULTS,
        exist_ok=True)

    # Load training data
    print("\nLoading Azure dataset...")
    train_data = load_filtered_data()

    # Dimensions
    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    print(f"\nTASCAR Configuration:")
    print(f"  State dim:       {state_dim}")
    print(f"  Action dim:      {action_dim}")
    print(f"  Sequence length: "
          f"{SEQUENCE_LENGTH}")
    print(f"  Transformer dim: "
          f"{TRANSFORMER_DIM}")
    print(f"  Episodes:        "
          f"{TASCAR_EPISODES}")

    # Create Transformer encoder
    transformer = TransformerEncoder(
        state_dim)

    # Create SAC agent
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=transformer)

    # Logger
    logger        = TASCARLogger()
    calls_per_ep  = EVAL_CALLS
    current_theta = THETA

    print(f"\nStarting training...")
    print("=" * 55)

    for episode in range(
            1, TASCAR_EPISODES + 1):

        # Random start for diversity
        max_start = max(
            1,
            len(train_data) - calls_per_ep)
        start_idx = np.random.randint(
            0, max_start)
        episode_calls = train_data[
            start_idx:
            start_idx + calls_per_ep]

        if len(episode_calls) < 1000:
            episode_calls = (
                train_data[:calls_per_ep])

        # Fresh S-Cache each episode
        scache  = SCache()
        history = StateHistoryBuffer(
            SEQUENCE_LENGTH,
            state_dim)

        # Get initial state
        raw_state = np.array(
            scache.get_state(),
            dtype=np.float32)
        mean = np.mean(raw_state)
        std  = np.std(raw_state)
        if std > 0:
            raw_state = (
                (raw_state - mean) / std)
        history.add(raw_state)

        # Get encoded initial state
        seq           = history.get_sequence()
        encoded_state = (
            agent.get_encoded_state(seq))

        # Episode tracking
        ep_reward   = 0.0
        ep_cold     = 0
        ep_warm     = 0
        step_cold   = 0
        step_warm   = 0
        call_count  = 0
        wmt_before  = 0.0
        ep_actor_loss  = []
        ep_critic_loss = []

        for call in episode_calls:

            # Process call
            is_warm = scache.handle_request(
                call)

            if is_warm:
                step_warm += 1
                ep_warm   += 1
            else:
                step_cold += 1
                ep_cold   += 1

            call_count += 1

            # Agent decides every DELTA calls
            if call_count % DELTA == 0:

                # Get new raw state
                new_raw = np.array(
                    scache.get_state(),
                    dtype=np.float32)
                mean = np.mean(new_raw)
                std  = np.std(new_raw)
                if std > 0:
                    new_raw = (
                        (new_raw - mean) /
                        std)

                history.add(new_raw)
                new_seq = (
                    history.get_sequence())
                next_encoded = (
                    agent.get_encoded_state(
                        new_seq))

                # Dynamic theta
                total_step = (
                    step_cold + step_warm)
                cold_rate = (
                    step_cold /
                    total_step
                    if total_step > 0
                    else 0)

                current_theta = (
                    compute_dynamic_theta(
                        cold_rate,
                        current_theta))

                # Calculate reward
                current_wmt = (
                    scache
                    .get_total_wasted_memory_time())
                wmt_change = max(
                    0,
                    current_wmt - wmt_before)
                wmt_before = current_wmt

                r1_norm = min(
                    cold_rate, 1.0)
                r2_norm = min(
                    wmt_change / 100.0,
                    1.0)
                reward = -(
                    current_theta *
                    r1_norm +
                    (1 - current_theta) *
                    r2_norm)

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

                # Update SAC agent
                result = agent.update()
                if result[0] is not None:
                    ep_actor_loss.append(
                        result[0])
                    ep_critic_loss.append(
                        result[1])

                # Apply action to queues
                scales = (
                    agent.action_map[action])
                for q_idx, scale in (
                        enumerate(scales)):
                    if scale != 0:
                        scache.scale_queue(
                            q_idx, scale)

                # Reset step counters
                encoded_state = next_encoded
                step_cold = 0
                step_warm = 0

        # Episode complete
        total_calls = ep_cold + ep_warm
        cold_pct = (
            ep_cold / total_calls * 100
            if total_calls > 0 else 0)
        final_wmt = (
            scache
            .get_total_wasted_memory_time())

        avg_actor = (
            np.mean(ep_actor_loss)
            if ep_actor_loss else 0)
        avg_critic = (
            np.mean(ep_critic_loss)
            if ep_critic_loss else 0)

        # Log episode
        logger.log_episode(
            episode,
            ep_reward,
            cold_pct,
            final_wmt,
            current_theta,
            avg_actor,
            avg_critic)

        # Save best model
        if ep_reward > logger.best_reward:
            agent.save(
                TASCAR_MODEL_PATH +
                "best/")

        # Print progress
        if episode % 10 == 0:
            avg_r = np.mean(
                logger.rewards[-10:])
            avg_c = np.mean(
                logger.cold_rates[-10:])
            print(
                f"Ep {episode:3d} | "
                f"Reward: {avg_r:6.3f} | "
                f"Cold%: {avg_c:5.1f}% | "
                f"Theta: "
                f"{current_theta:.3f} | "
                f"Buffer: "
                f"{len(agent.buffer):5d}")

        # Checkpoint every 50 episodes
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
    print(f"Model saved to: "
          f"{TASCAR_MODEL_PATH}")
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
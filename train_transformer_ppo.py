# train_transformer_ppo.py
# Ablation Variant 3: Transformer + PPO
# PPO agent WITH Transformer!
# Sequence input like TASCAR!
# But PPO not SAC!
# Isolates Transformer contribution!

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
    TASCAR_DELTA,
    THETA,
    THETA_MIN,
    THETA_MAX,
    THETA_ADAPT_RATE,
    SEQUENCE_LENGTH,
    TRANSFORMER_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    TRANSFORMER_FF_DIM,
    DROPOUT_RATE,
    LEARNING_RATE_ACTOR,
    LEARNING_RATE_CRITIC,
    DISCOUNT_FACTOR,
    GAE_LAMBDA,
    PPO_CLIP,
    ENTROPY_COEFF,
    MINI_BATCH_SIZE,
    EPOCHS_PER_UPDATE,
    SCALING_FACTOR,
    CONVERGENCE_WINDOW,
    CONVERGENCE_THRESHOLD,
    RANDOM_SEED,
    TRANSFORMER_PPO_MODEL_PATH,
    ABLATION_RESULTS,
    ABLATION_EPISODES)
from simulator import AzureDataLoader
from scache import SCache
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ─────────────────────────────────────────
# TRANSFORMER PPO NETWORKS
# Transformer encoder + PPO!
# No SAC! No replay buffer!
# On-policy like CASR but with
# Transformer state encoding!
# ─────────────────────────────────────────

class TransformerPPOActor(nn.Module):
    """
    PPO Actor with Transformer input!
    Input: 64-dim encoded state!
    Output: action probabilities!
    """
    def __init__(self,
                 encoded_dim,
                 action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim))

    def forward(self, x):
        return F.softmax(
            self.net(x), dim=-1)


class TransformerPPOCritic(nn.Module):
    """
    PPO Critic with Transformer input!
    Input: 64-dim encoded state!
    Output: state value!
    """
    def __init__(self,
                 encoded_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x)


class TransformerPPOAgent:
    """
    PPO Agent with Transformer encoder!
    Ablation V3: Transformer contribution!
    Uses sequence of states like TASCAR!
    But PPO updates not SAC!
    On-policy: no replay buffer!
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 transformer):
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.transformer = transformer
        self.device      = torch.device(
            'cpu')

        encoded_dim = TRANSFORMER_DIM

        self.actor  = TransformerPPOActor(
            encoded_dim,
            action_dim).to(self.device)
        self.critic = (
            TransformerPPOCritic(
                encoded_dim
            ).to(self.device))

        self.actor_opt  = optim.Adam(
            list(self.actor.parameters()) +
            list(self.transformer
                 .parameters()),
            lr=LEARNING_RATE_ACTOR)
        self.critic_opt = optim.Adam(
            self.critic.parameters(),
            lr=LEARNING_RATE_CRITIC)

        # PPO buffer
        self.states      = []
        self.actions     = []
        self.rewards     = []
        self.log_probs   = []
        self.values      = []
        self.dones       = []

        # Action map same as TASCAR!
        self.action_map = {}
        choices = [
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
            self.action_map[i] = action

    def get_encoded_state(self, seq):
        """
        Encode sequence via Transformer!
        Same as TASCAR encoding!
        """
        with torch.no_grad():
            seq_t = torch.FloatTensor(
                seq).unsqueeze(0).to(
                self.device)
            encoded = self.transformer(
                seq_t)
            return (encoded
                    .squeeze(0)
                    .cpu()
                    .numpy())

    def choose_action(self,
                      encoded_state,
                      evaluate=False):
        with torch.no_grad():
            s = torch.FloatTensor(
                encoded_state
            ).unsqueeze(0).to(
                self.device)
            probs = self.actor(s)
            value = self.critic(s)
            if evaluate:
                action   = probs.argmax(
                    dim=-1).item()
                log_prob = torch.log(
                    probs[0, action] +
                    1e-8).item()
            else:
                dist     = (
                    torch.distributions
                    .Categorical(probs))
                action   = dist.sample(
                ).item()
                log_prob = dist.log_prob(
                    torch.tensor(action)
                ).item()
        return (action,
                log_prob,
                value.item())

    def store_experience(self,
                         state,
                         action,
                         reward,
                         log_prob,
                         value,
                         done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        if len(self.states) < (
                MINI_BATCH_SIZE):
            return None, None

        states    = torch.FloatTensor(
            np.array(self.states)
        ).to(self.device)
        actions   = torch.LongTensor(
            self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(
            self.log_probs).to(self.device)
        old_values    = torch.FloatTensor(
            self.values).to(self.device)

        # Compute returns and advantages
        returns    = []
        advantages = []
        R          = 0.0
        adv        = 0.0
        next_value = 0.0

        for i in reversed(
                range(len(self.rewards))):
            R   = (self.rewards[i] +
                   DISCOUNT_FACTOR *
                   R *
                   (1 - self.dones[i]))
            td  = (self.rewards[i] +
                   DISCOUNT_FACTOR *
                   next_value *
                   (1 - self.dones[i]) -
                   self.values[i])
            adv = (td +
                   DISCOUNT_FACTOR *
                   GAE_LAMBDA *
                   adv *
                   (1 - self.dones[i]))
            returns.insert(0, R)
            advantages.insert(0, adv)
            next_value = self.values[i]

        returns    = torch.FloatTensor(
            returns).to(self.device)
        advantages = torch.FloatTensor(
            advantages).to(self.device)
        advantages = (
            (advantages -
             advantages.mean()) /
            (advantages.std() + 1e-8))

        total_actor_loss  = 0.0
        total_critic_loss = 0.0

        for _ in range(
                EPOCHS_PER_UPDATE):
            idx = torch.randperm(
                len(states))
            for start in range(
                    0, len(states),
                    MINI_BATCH_SIZE):
                batch_idx = idx[
                    start:start +
                    MINI_BATCH_SIZE]
                if len(batch_idx) < 2:
                    continue

                b_states     = states[
                    batch_idx]
                b_actions    = actions[
                    batch_idx]
                b_old_lp     = old_log_probs[
                    batch_idx]
                b_returns    = returns[
                    batch_idx]
                b_advantages = advantages[
                    batch_idx]

                probs    = self.actor(
                    b_states)
                dist     = (
                    torch.distributions
                    .Categorical(probs))
                new_lp   = dist.log_prob(
                    b_actions)
                entropy  = dist.entropy()
                values   = self.critic(
                    b_states).squeeze(1)

                ratio    = torch.exp(
                    new_lp - b_old_lp)
                surr1    = (
                    ratio * b_advantages)
                surr2    = (
                    torch.clamp(
                        ratio,
                        1 - PPO_CLIP,
                        1 + PPO_CLIP) *
                    b_advantages)
                a_loss   = (
                    -torch.min(
                        surr1, surr2
                    ).mean() -
                    ENTROPY_COEFF *
                    entropy.mean())
                c_loss   = F.mse_loss(
                    values, b_returns)

                self.actor_opt.zero_grad()
                a_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    0.5)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                c_loss.backward()
                self.critic_opt.step()

                total_actor_loss  += (
                    a_loss.item())
                total_critic_loss += (
                    c_loss.item())

        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.log_probs = []
        self.values    = []
        self.dones     = []

        return (total_actor_loss,
                total_critic_loss)

    def save(self, path):
        os.makedirs(
            path, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            path + 'actor.pth')
        torch.save(
            self.critic.state_dict(),
            path + 'critic.pth')
        torch.save(
            self.transformer
            .state_dict(),
            path + 'transformer.pth')
        print(f"Saved to {path}")

    def load(self, path):
        self.actor.load_state_dict(
            torch.load(
                path + 'actor.pth',
                map_location='cpu'))
        self.critic.load_state_dict(
            torch.load(
                path + 'critic.pth',
                map_location='cpu'))
        self.transformer.load_state_dict(
            torch.load(
                path + 'transformer.pth',
                map_location='cpu'))
        print(
            f"Loaded from {path}")


# ─────────────────────────────────────────
# HELPER FUNCTIONS
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


def compute_dynamic_theta(
        cold_start_rate,
        current_theta):
    if cold_start_rate > 0.95:
        return min(
            current_theta +
            THETA_ADAPT_RATE,
            THETA_MAX)
    elif cold_start_rate < 0.85:
        return max(
            current_theta -
            THETA_ADAPT_RATE,
            THETA_MIN)
    return current_theta


class RewardNormalizer:
    def __init__(self):
        self.r1_min =  float('inf')
        self.r1_max = -float('inf')
        self.r2_min =  float('inf')
        self.r2_max = -float('inf')

    def calculate(self, cold_starts,
                  wmt_change, theta):
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
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────

def train_transformer_ppo():
    """
    Train Transformer+PPO ablation!
    Has Transformer encoder!
    Has sequence input!
    But PPO not SAC!
    No replay buffer!
    On-policy updates!
    """
    os.makedirs(
        TRANSFORMER_PPO_MODEL_PATH,
        exist_ok=True)
    os.makedirs(
        ABLATION_RESULTS,
        exist_ok=True)

    np.random.seed(RANDOM_SEED)
    print(
        "Transformer+PPO Ablation!")
    print(
        f"Seed: {RANDOM_SEED}")
    print(
        "Has Transformer!")
    print(
        "Uses PPO not SAC!")

    print("\nLoading dataset...")
    train_data = load_filtered_data()

    state_dim    = NUM_QUEUES * 7
    action_dim   = 3 ** NUM_QUEUES
    steps_per_ep = (
        EVAL_CALLS // TASCAR_DELTA)

    print(f"\nTransformer+PPO Config:")
    print(f"  State dim:     {state_dim}")
    print(f"  Encoded dim:   {TRANSFORMER_DIM}")
    print(f"  Action dim:    {action_dim}")
    print(f"  Seq length:    {SEQUENCE_LENGTH}")
    print(f"  Steps/episode: {steps_per_ep}")
    print(f"  Episodes:      {ABLATION_EPISODES}")
    print(f"  HAS Transformer!")
    print(f"  PPO not SAC!")

    transformer = TransformerEncoder(
        state_dim)
    agent = TransformerPPOAgent(
        state_dim,
        action_dim,
        transformer)

    reward_norm   = RewardNormalizer()
    best_reward   = float('-inf')
    current_theta = THETA
    start_time    = time.time()

    rewards_log    = []
    cold_rates_log = []
    episodes_log   = []
    convergence_ep = -1

    print(
        f"\nStarting "
        f"Transformer+PPO training...")
    print("=" * 50)

    for episode in range(
            1, ABLATION_EPISODES + 1):

        max_start = max(
            1,
            len(train_data) -
            EVAL_CALLS)
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

        raw_state = normalize_state(
            scache.get_state())
        history.add(raw_state)
        seq = history.get_sequence()
        encoded_state = (
            agent.get_encoded_state(seq))

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

                total = (
                    step_cold + step_warm)
                cold_rate = (
                    step_cold / total
                    if total > 0 else 0)
                current_theta = (
                    compute_dynamic_theta(
                        cold_rate,
                        current_theta))

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
                        current_theta))

                (action,
                 log_prob,
                 value) = (
                    agent.choose_action(
                        encoded_state))

                agent.store_experience(
                    encoded_state,
                    action,
                    reward,
                    log_prob,
                    value,
                    False)

                ep_reward  += reward
                steps_done += 1

                scales = (
                    agent.action_map[
                        action])
                for q_idx, scale in (
                        enumerate(
                            scales)):
                    if scale != 0:
                        scache.scale_queue(
                            q_idx, scale)

                encoded_state = (
                    next_encoded)
                step_cold = 0
                step_warm = 0

        # PPO update at end of episode
        agent.update()

        total_calls = ep_cold + ep_warm
        cold_pct = (
            ep_cold / total_calls * 100
            if total_calls > 0 else 0)

        avg_ep_reward = (
            ep_reward / steps_done
            if steps_done > 0
            else ep_reward)

        rewards_log.append(
            avg_ep_reward)
        cold_rates_log.append(cold_pct)
        episodes_log.append(episode)

        if avg_ep_reward > best_reward:
            best_reward = avg_ep_reward
            agent.save(
                TRANSFORMER_PPO_MODEL_PATH +
                "best/")

        if episode % 50 == 0:
            agent.save(
                TRANSFORMER_PPO_MODEL_PATH +
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

    elapsed = time.time() - start_time
    agent.save(
        TRANSFORMER_PPO_MODEL_PATH +
        "best/")

    logs = {
        'variant':
            'Transformer-PPO',
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
        'has_transformer': True,
        'has_sac':         False,
        'has_dynamic_theta': True,
    }
    log_path = (
        ABLATION_RESULTS +
        'transformer_ppo_logs.json')
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

    print("\n" + "=" * 50)
    print("Transformer+PPO Complete!")
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
        f"Logs saved:     "
        f"{log_path}")

    return agent


if __name__ == "__main__":
    print("=" * 50)
    print("Ablation V3: Transformer+PPO")
    print("Transformer WITH PPO!")
    print("No SAC! No replay buffer!")
    print(f"Seed: {RANDOM_SEED}")
    print(
        f"Episodes: "
        f"{ABLATION_EPISODES}")
    print("=" * 50)
    train_transformer_ppo()
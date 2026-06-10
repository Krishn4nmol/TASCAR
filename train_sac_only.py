# train_sac_only.py
# Ablation Variant 2: SAC Only
# SAC agent WITHOUT Transformer!
# Uses raw 21-dim state directly!
# No sequence, no attention!
# Isolates SAC contribution!

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
    TASCAR_EPISODES,
    SAC_BATCH_SIZE,
    SAC_UPDATES_PER_STEP,
    SAC_BUFFER_SIZE,
    SAC_LR_ACTOR,
    SAC_LR_CRITIC,
    SAC_LR_ALPHA,
    SAC_GAMMA,
    SAC_TAU,
    TARGET_ENTROPY,
    AUTO_ENTROPY,
    SCALING_FACTOR,
    CONVERGENCE_WINDOW,
    CONVERGENCE_THRESHOLD,
    RANDOM_SEED,
    SAC_ONLY_MODEL_PATH,
    ABLATION_RESULTS,
    ABLATION_EPISODES)
from simulator import AzureDataLoader
from scache import SCache

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


# ─────────────────────────────────────────
# SAC ONLY NETWORKS
# No Transformer! Raw state input!
# State dim = 21 directly!
# ─────────────────────────────────────────

class SACOnlyActor(nn.Module):
    """
    Actor for SAC-Only variant.
    Input: raw 21-dim state!
    No Transformer encoding!
    """
    def __init__(self,
                 state_dim,
                 action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim))

    def forward(self, x):
        return F.softmax(
            self.net(x), dim=-1)


class SACOnlyCritic(nn.Module):
    """
    Critic for SAC-Only variant.
    Input: raw 21-dim state!
    No Transformer encoding!
    """
    def __init__(self,
                 state_dim,
                 action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim))

    def forward(self, x):
        return self.net(x)


class SACOnlyAgent:
    """
    SAC Agent using raw state!
    No Transformer!
    Ablation V2: SAC contribution only!
    """
    def __init__(self,
                 state_dim,
                 action_dim):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = torch.device(
            'cpu')

        self.actor    = SACOnlyActor(
            state_dim,
            action_dim).to(self.device)
        self.critic1  = SACOnlyCritic(
            state_dim,
            action_dim).to(self.device)
        self.critic2  = SACOnlyCritic(
            state_dim,
            action_dim).to(self.device)
        self.critic1_target = (
            SACOnlyCritic(
                state_dim,
                action_dim
            ).to(self.device))
        self.critic2_target = (
            SACOnlyCritic(
                state_dim,
                action_dim
            ).to(self.device))

        self.critic1_target.load_state_dict(
            self.critic1.state_dict())
        self.critic2_target.load_state_dict(
            self.critic2.state_dict())

        self.actor_opt   = optim.Adam(
            self.actor.parameters(),
            lr=SAC_LR_ACTOR)
        self.critic1_opt = optim.Adam(
            self.critic1.parameters(),
            lr=SAC_LR_CRITIC)
        self.critic2_opt = optim.Adam(
            self.critic2.parameters(),
            lr=SAC_LR_CRITIC)

        self.log_alpha = torch.tensor(
            0.0, requires_grad=True,
            device=self.device)
        self.alpha_opt = optim.Adam(
            [self.log_alpha],
            lr=SAC_LR_ALPHA)
        self.target_entropy = (
            TARGET_ENTROPY)

        self.buffer = deque(
            maxlen=SAC_BUFFER_SIZE)

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

    def choose_action(self,
                      state,
                      evaluate=False):
        with torch.no_grad():
            s = torch.FloatTensor(
                state).unsqueeze(0).to(
                self.device)
            probs  = self.actor(s)
            if evaluate:
                action = probs.argmax(
                    dim=-1).item()
            else:
                action = torch.multinomial(
                    probs, 1).item()
        return action

    def store_experience(self,
                         state,
                         action,
                         reward,
                         next_state,
                         done):
        self.buffer.append((
            state, action,
            reward, next_state,
            done))

    def update(self):
        if len(self.buffer) < (
                SAC_BATCH_SIZE):
            return None, None

        batch = random.sample(
            self.buffer,
            SAC_BATCH_SIZE)
        (states, actions,
         rewards, next_states,
         dones) = zip(*batch)

        states      = torch.FloatTensor(
            np.array(states)
        ).to(self.device)
        actions     = torch.LongTensor(
            actions).to(self.device)
        rewards     = torch.FloatTensor(
            rewards).to(self.device)
        next_states = torch.FloatTensor(
            np.array(next_states)
        ).to(self.device)
        dones       = torch.FloatTensor(
            dones).to(self.device)

        alpha = self.log_alpha.exp(
        ).detach()

        with torch.no_grad():
            next_probs = self.actor(
                next_states)
            next_log_probs = torch.log(
                next_probs + 1e-8)
            next_q1 = self.critic1_target(
                next_states)
            next_q2 = self.critic2_target(
                next_states)
            next_q  = torch.min(
                next_q1, next_q2)
            next_v  = (next_probs * (
                next_q - alpha *
                next_log_probs)).sum(
                dim=-1)
            target_q = (
                rewards +
                SAC_GAMMA *
                (1 - dones) * next_v)

        curr_q1 = self.critic1(
            states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        curr_q2 = self.critic2(
            states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        c1_loss = F.mse_loss(
            curr_q1, target_q)
        c2_loss = F.mse_loss(
            curr_q2, target_q)

        self.critic1_opt.zero_grad()
        c1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        c2_loss.backward()
        self.critic2_opt.step()

        probs     = self.actor(states)
        log_probs = torch.log(
            probs + 1e-8)
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q  = torch.min(q1, q2)
        actor_loss = (probs * (
            alpha * log_probs - q)
        ).sum(dim=-1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if AUTO_ENTROPY:
            entropy = -(
                probs *
                log_probs).sum(
                dim=-1).mean()
            alpha_loss = (
                self.log_alpha * (
                    entropy -
                    self.target_entropy
                ).detach())
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        for p, tp in zip(
                self.critic1.parameters(),
                self.critic1_target.parameters()):
            tp.data.copy_(
                SAC_TAU * p.data +
                (1 - SAC_TAU) * tp.data)
        for p, tp in zip(
                self.critic2.parameters(),
                self.critic2_target.parameters()):
            tp.data.copy_(
                SAC_TAU * p.data +
                (1 - SAC_TAU) * tp.data)

        return (actor_loss.item(),
                c1_loss.item())

    def save(self, path):
        os.makedirs(
            path, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            path + 'actor.pth')
        torch.save(
            self.critic1.state_dict(),
            path + 'critic1.pth')
        torch.save(
            self.critic2.state_dict(),
            path + 'critic2.pth')
        print(f"Saved to {path}")

    def load(self, path):
        self.actor.load_state_dict(
            torch.load(
                path + 'actor.pth',
                map_location='cpu'))
        self.critic1.load_state_dict(
            torch.load(
                path + 'critic1.pth',
                map_location='cpu'))
        self.critic2.load_state_dict(
            torch.load(
                path + 'critic2.pth',
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
        f"  Total after filter: "
        f"{len(train_data)}")
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
# WARMUP BUFFER
# ─────────────────────────────────────────

def warmup_buffer(agent,
                  train_data,
                  state_dim,
                  warmup_episodes=20):
    print(
        f"\nWarming up SAC-Only "
        f"buffer...")
    for ep in range(warmup_episodes):
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

        scache     = SCache()
        call_count = 0
        wmt_before = 0.0
        step_cold  = 0
        step_warm  = 0

        raw = normalize_state(
            scache.get_state())

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
                new_raw = (
                    normalize_state(
                        scache
                        .get_state()))
                action = (
                    np.random.randint(
                        0,
                        agent.action_dim))
                total = (
                    step_cold +
                    step_warm)
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
                    min(wmt_change /
                        100.0, 1.0))
                if (not np.isnan(
                        raw).any() and
                        not np.isnan(
                            new_raw
                        ).any()):
                    agent.store_experience(
                        raw, action,
                        reward, new_raw,
                        False)
                scales = (
                    agent.action_map[
                        action])
                for q_idx, scale in (
                        enumerate(
                            scales)):
                    if scale != 0:
                        scache.scale_queue(
                            q_idx, scale)
                raw       = new_raw
                step_cold = 0
                step_warm = 0

        print(
            f"  Warmup ep "
            f"{ep+1:2d}: "
            f"Buffer: "
            f"{len(agent.buffer)}")
    print(
        f"Warmup complete! "
        f"Buffer: {len(agent.buffer)}")


# ─────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────

def train_sac_only():
    """
    Train SAC-Only ablation variant!
    No Transformer!
    Raw 21-dim state directly!
    """
    os.makedirs(
        SAC_ONLY_MODEL_PATH,
        exist_ok=True)
    os.makedirs(
        ABLATION_RESULTS,
        exist_ok=True)

    np.random.seed(RANDOM_SEED)
    print(
        f"SAC-Only Ablation Training")
    print(
        f"Random seed: {RANDOM_SEED}")
    print(
        f"No Transformer! "
        f"Raw state only!")

    print("\nLoading Azure dataset...")
    train_data = load_filtered_data()

    state_dim    = NUM_QUEUES * 7
    action_dim   = 3 ** NUM_QUEUES
    steps_per_ep = (
        EVAL_CALLS // TASCAR_DELTA)

    print(f"\nSAC-Only Configuration:")
    print(f"  State dim:     {state_dim}")
    print(f"  Action dim:    {action_dim}")
    print(f"  Delta:         {TASCAR_DELTA}")
    print(f"  Steps/episode: {steps_per_ep}")
    print(f"  Episodes:      {ABLATION_EPISODES}")
    print(f"  NO Transformer!")
    print(f"  NO sequence input!")

    agent = SACOnlyAgent(
        state_dim, action_dim)

    warmup_buffer(
        agent, train_data,
        state_dim,
        warmup_episodes=20)

    reward_norm   = RewardNormalizer()
    best_reward   = float('-inf')
    current_theta = THETA
    start_time    = time.time()

    rewards_log     = []
    cold_rates_log  = []
    episodes_log    = []
    convergence_ep  = -1

    print(f"\nStarting SAC-Only training...")
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

            if (call_count %
                    TASCAR_DELTA == 0):
                new_raw = (
                    normalize_state(
                        scache
                        .get_state()))

                total = (
                    step_cold +
                    step_warm)
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

                action = (
                    agent.choose_action(
                        raw_state))

                if (not np.isnan(
                        raw_state
                    ).any() and
                        not np.isnan(
                            new_raw
                        ).any()):
                    agent.store_experience(
                        raw_state,
                        action,
                        reward,
                        new_raw,
                        False)

                ep_reward  += reward
                steps_done += 1

                for _ in range(
                        SAC_UPDATES_PER_STEP):
                    agent.update()

                scales = (
                    agent.action_map[
                        action])
                for q_idx, scale in (
                        enumerate(
                            scales)):
                    if scale != 0:
                        scache.scale_queue(
                            q_idx, scale)

                raw_state = new_raw
                step_cold = 0
                step_warm = 0

        total_calls = (
            ep_cold + ep_warm)
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
                SAC_ONLY_MODEL_PATH +
                "best/")

        if episode % 50 == 0:
            agent.save(
                SAC_ONLY_MODEL_PATH +
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
                f"Buffer: "
                f"{len(agent.buffer):5d} | "
                f"Time: {elapsed:.0f}s")

    elapsed = time.time() - start_time
    agent.save(
        SAC_ONLY_MODEL_PATH + "best/")

    logs = {
        'variant':        'SAC-Only',
        'episodes':       episodes_log,
        'rewards':        rewards_log,
        'cold_start_rates':
            cold_rates_log,
        'best_reward':    best_reward,
        'training_time':  elapsed,
        'convergence_ep': convergence_ep,
        'random_seed':    RANDOM_SEED,
        'has_transformer': False,
        'has_sac':         True,
        'has_dynamic_theta': True,
    }
    log_path = (
        ABLATION_RESULTS +
        'sac_only_logs.json')
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

    print("\n" + "=" * 50)
    print("SAC-Only Training Complete!")
    print("=" * 50)
    print(f"Best reward:    {best_reward:.4f}")
    print(f"Training time:  {elapsed:.1f}s")
    print(f"Convergence ep: {convergence_ep}")
    print(f"Logs saved:     {log_path}")

    return agent


if __name__ == "__main__":
    print("=" * 50)
    print("Ablation V2: SAC-Only")
    print("SAC WITHOUT Transformer!")
    print("Raw 21-dim state!")
    print(f"Seed: {RANDOM_SEED}")
    print(f"Episodes: {ABLATION_EPISODES}")
    print("=" * 50)
    train_sac_only()
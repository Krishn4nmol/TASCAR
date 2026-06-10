# sac_agent.py
# Soft Actor-Critic Agent for TASCAR
# Fixed with gradient clipping
# and NaN protection!

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from config import (
    SAC_ALPHA,
    SAC_TAU,
    SAC_GAMMA,
    SAC_LR_ACTOR,
    SAC_LR_CRITIC,
    SAC_LR_ALPHA,
    SAC_BUFFER_SIZE,
    SAC_BATCH_SIZE,
    AUTO_ENTROPY,
    TARGET_ENTROPY,
    HIDDEN_LAYER_SIZE,
    TRANSFORMER_DIM,
    TASCAR_MODEL_PATH,
    NUM_QUEUES,
    SCALING_FACTOR
)


# ─────────────────────────────────────────
# REPLAY BUFFER
# Off-policy buffer
# SAC reuses old experiences!
# More sample efficient than PPO!
# ─────────────────────────────────────────

class ReplayBuffer:
    """
    Off-policy replay buffer.

    Key difference from PPO:
    PPO clears buffer after update
    SAC KEEPS all experiences!
    Randomly samples from all past data!

    More sample efficient!
    """
    def __init__(self, capacity):
        self.buffer = deque(
            maxlen=capacity)

    def push(self,
             state,
             action,
             reward,
             next_state,
             done):
        self.buffer.append((
            np.array(
                state,
                dtype=np.float32),
            int(action),
            float(reward),
            np.array(
                next_state,
                dtype=np.float32),
            float(done)))

    def sample(self, batch_size):
        batch = random.sample(
            self.buffer,
            batch_size)
        (states,
         actions,
         rewards,
         next_states,
         dones) = zip(*batch)

        return (
            torch.FloatTensor(
                np.array(states)),
            torch.LongTensor(
                np.array(actions)),
            torch.FloatTensor(
                np.array(rewards)),
            torch.FloatTensor(
                np.array(next_states)),
            torch.FloatTensor(
                np.array(dones)))

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        return len(
            self.buffer) >= batch_size


# ─────────────────────────────────────────
# ACTOR NETWORK
# Picks which action to take
# Input = Transformer encoded state
# Output = probability per action
# ─────────────────────────────────────────

class SACActorNetwork(nn.Module):
    """
    Actor network for SAC.

    Takes Transformer encoded state!
    Key difference from CASR PPO:

    CASR: raw state → PPO actor
    TASCAR: raw state → Transformer
            → enriched → SAC actor
    """
    def __init__(self,
                 state_dim,
                 action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                state_dim,
                HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(
                HIDDEN_LAYER_SIZE,
                HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(
                HIDDEN_LAYER_SIZE,
                action_dim))

    def forward(self, state):
        logits = self.network(state)
        # Clamp to prevent NaN!
        logits = torch.clamp(
            logits, -10, 10)
        probs  = F.softmax(
            logits, dim=-1)
        # Add small epsilon
        # to prevent zero probs!
        probs  = probs + 1e-8
        probs  = probs / probs.sum(
            dim=-1, keepdim=True)
        return probs


# ─────────────────────────────────────────
# CRITIC NETWORK
# Estimates value of state
# SAC uses TWO critics!
# Takes minimum to reduce bias!
# ─────────────────────────────────────────

class SACCriticNetwork(nn.Module):
    """
    Critic network for SAC.

    KEY DIFFERENCE FROM PPO:
    PPO uses ONE critic
    SAC uses TWO critics!

    Two critics → take minimum →
    Reduces overestimation bias!
    More stable training!
    """
    def __init__(self,
                 state_dim,
                 action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                state_dim,
                HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(
                HIDDEN_LAYER_SIZE,
                HIDDEN_LAYER_SIZE),
            nn.ReLU(),
            nn.Linear(
                HIDDEN_LAYER_SIZE,
                action_dim))

    def forward(self, state):
        return self.network(state)


# ─────────────────────────────────────────
# SAC AGENT
# Complete Soft Actor-Critic Agent
# Works with Transformer encoder!
# Fixed with gradient clipping!
# ─────────────────────────────────────────

class SACAgent:
    """
    Complete SAC Agent.

    Key improvements over PPO in CASR:
    1. TWO critics reduce overestimation
    2. Entropy temperature encourages
       better exploration
    3. Off-policy learning reuses
       old experiences efficiently
    4. Automatic alpha tuning
    5. Works with Transformer encoder
    6. Gradient clipping prevents NaN!
    """
    def __init__(self,
                 transformer_dim,
                 action_dim,
                 transformer):

        self.action_dim  = action_dim
        self.transformer = transformer
        state_dim        = transformer_dim

        # Actor network
        self.actor = SACActorNetwork(
            state_dim, action_dim)

        # TWO critic networks!
        self.critic1 = SACCriticNetwork(
            state_dim, action_dim)
        self.critic2 = SACCriticNetwork(
            state_dim, action_dim)

        # Target critics for stability
        self.target_critic1 = (
            SACCriticNetwork(
                state_dim, action_dim))
        self.target_critic2 = (
            SACCriticNetwork(
                state_dim, action_dim))

        # Copy weights to targets
        self._update_targets(tau=1.0)

        # Entropy temperature alpha
        if AUTO_ENTROPY:
            self.log_alpha = torch.zeros(
                1, requires_grad=True)
            self.alpha = (
                self.log_alpha
                .exp().item())
            self.alpha_optimizer = (
                optim.Adam(
                    [self.log_alpha],
                    lr=SAC_LR_ALPHA))
        else:
            self.alpha           = SAC_ALPHA
            self.log_alpha       = None
            self.alpha_optimizer = None

        # Optimizers with lower LR
        # for stability!
        self.actor_optimizer = (
            optim.Adam(
                self.actor.parameters(),
                lr=SAC_LR_ACTOR))
        self.critic1_optimizer = (
            optim.Adam(
                self.critic1.parameters(),
                lr=SAC_LR_CRITIC))
        self.critic2_optimizer = (
            optim.Adam(
                self.critic2.parameters(),
                lr=SAC_LR_CRITIC))

        # Replay buffer
        self.buffer = ReplayBuffer(
            SAC_BUFFER_SIZE)

        # Action map same as CASR!
        self.action_map = (
            self._build_action_map())

    def _build_action_map(self):
        """Same action map as CASR!"""
        choices    = [
            -SCALING_FACTOR,
            0,
            SCALING_FACTOR]
        action_map = {}
        for i in range(
                3 ** NUM_QUEUES):
            action = []
            temp   = i
            for _ in range(NUM_QUEUES):
                action.append(
                    choices[temp % 3])
                temp //= 3
            action_map[i] = action
        return action_map

    def get_encoded_state(
            self, state_sequence):
        """
        Pass state sequence through
        Transformer to get rich state!

        This is the KEY step that
        makes TASCAR different from CASR!
        """
        with torch.no_grad():
            seq_tensor = (
                torch.FloatTensor(
                    state_sequence)
                .unsqueeze(0))
            encoded = self.transformer(
                seq_tensor)

        result = (
            encoded.squeeze(0).numpy())

        # Check for NaN in encoding!
        if np.isnan(result).any():
            result = np.zeros_like(result)

        return result

    def choose_action(
            self,
            encoded_state,
            evaluate=False):
        """
        Pick action from encoded state.

        Training: sample randomly
        Evaluation: pick best action

        NaN protection added!
        Returns random action if NaN!
        """
        state_tensor = (
            torch.FloatTensor(
                encoded_state)
            .unsqueeze(0))

        with torch.no_grad():
            probs = self.actor(
                state_tensor)

        # NaN protection!
        if torch.isnan(probs).any():
            return np.random.randint(
                0, self.action_dim)

        if evaluate:
            # Best action for evaluation
            action = probs.argmax(
                dim=-1).item()
        else:
            # Sample for exploration
            dist = (
                torch.distributions
                .Categorical(probs))
            action = dist.sample().item()

        return action

    def store_experience(
            self,
            encoded_state,
            action,
            reward,
            next_encoded_state,
            done):
        """Store experience in buffer"""
        self.buffer.push(
            encoded_state,
            action,
            reward,
            next_encoded_state,
            done)

    def update(self):
        """
        SAC update step with
        gradient clipping!

        Updates:
        1. Both critics
        2. Actor
        3. Entropy temperature
        4. Target networks
        """
        if not self.buffer.is_ready(
                SAC_BATCH_SIZE):
            return None, None, None

        # Sample random batch
        (states,
         actions,
         rewards,
         next_states,
         dones) = self.buffer.sample(
            SAC_BATCH_SIZE)

        # ============================
        # UPDATE CRITICS
        # ============================
        with torch.no_grad():
            next_probs = self.actor(
                next_states)

            # NaN check!
            if torch.isnan(
                    next_probs).any():
                return None, None, None

            next_log_probs = torch.log(
                next_probs + 1e-8)

            # Take MINIMUM of two critics!
            target_q1 = (
                self.target_critic1(
                    next_states))
            target_q2 = (
                self.target_critic2(
                    next_states))
            target_q  = torch.min(
                target_q1, target_q2)

            # Entropy bonus
            target_v = (
                next_probs *
                (target_q -
                 self.alpha *
                 next_log_probs)
            ).sum(
                dim=-1,
                keepdim=True)

            # Bellman target
            target = (
                rewards.unsqueeze(-1) +
                SAC_GAMMA *
                (1 -
                 dones.unsqueeze(-1)) *
                target_v)

        # Current Q values
        current_q1 = (
            self.critic1(states)
            .gather(
                1,
                actions.unsqueeze(-1)))
        current_q2 = (
            self.critic2(states)
            .gather(
                1,
                actions.unsqueeze(-1)))

        # Critic losses
        critic1_loss = F.mse_loss(
            current_q1, target)
        critic2_loss = F.mse_loss(
            current_q2, target)

        # NaN check on losses!
        if (torch.isnan(critic1_loss)
                or torch.isnan(
                    critic2_loss)):
            return None, None, None

        # Update critic 1
        # WITH gradient clipping!
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic1.parameters(),
            max_norm=1.0)
        self.critic1_optimizer.step()

        # Update critic 2
        # WITH gradient clipping!
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic2.parameters(),
            max_norm=1.0)
        self.critic2_optimizer.step()

        # ============================
        # UPDATE ACTOR
        # ============================
        probs = self.actor(states)

        # NaN check!
        if torch.isnan(probs).any():
            return None, None, None

        log_probs = torch.log(
            probs + 1e-8)

        with torch.no_grad():
            q1 = self.critic1(states)
            q2 = self.critic2(states)
            q  = torch.min(q1, q2)

        # Actor loss with entropy
        actor_loss = (
            probs *
            (self.alpha *
             log_probs - q)
        ).sum(dim=-1).mean()

        # NaN check!
        if torch.isnan(actor_loss):
            return None, None, None

        # Update actor
        # WITH gradient clipping!
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=1.0)
        self.actor_optimizer.step()

        # ============================
        # UPDATE ALPHA
        # ============================
        alpha_loss = None
        if (AUTO_ENTROPY and
                self.log_alpha
                is not None):
            with torch.no_grad():
                probs = self.actor(
                    states)
                log_probs = torch.log(
                    probs + 1e-8)
                entropy = -(
                    probs * log_probs
                ).sum(dim=-1).mean()

            alpha_loss = -(
                self.log_alpha *
                (entropy +
                 TARGET_ENTROPY)
                .detach()).mean()

            if not torch.isnan(
                    alpha_loss):
                self.alpha_optimizer\
                    .zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer\
                    .step()
                self.alpha = (
                    self.log_alpha
                    .exp().item())

        # ============================
        # SOFT UPDATE TARGETS
        # ============================
        self._update_targets(
            tau=SAC_TAU)

        return (
            actor_loss.item(),
            critic1_loss.item(),
            alpha_loss.item()
            if alpha_loss is not None
            else 0.0)

    def _update_targets(self, tau):
        """
        Soft update target networks.
        Slowly moves targets toward
        current critic weights.
        Keeps training stable!
        """
        for target, current in zip(
            self.target_critic1
                .parameters(),
            self.critic1
                .parameters()):
            target.data.copy_(
                tau * current.data +
                (1 - tau) *
                target.data)

        for target, current in zip(
            self.target_critic2
                .parameters(),
            self.critic2
                .parameters()):
            target.data.copy_(
                tau * current.data +
                (1 - tau) *
                target.data)

    def save(self, path):
        """Save all model weights"""
        os.makedirs(
            path, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            path + "actor.pth")
        torch.save(
            self.critic1.state_dict(),
            path + "critic1.pth")
        torch.save(
            self.critic2.state_dict(),
            path + "critic2.pth")
        torch.save(
            self.transformer
                .state_dict(),
            path + "transformer.pth")
        print(f"Saved to {path}")

    def load(self, path):
        """Load all model weights"""
        self.actor.load_state_dict(
            torch.load(
                path + "actor.pth",
                map_location = 'cpu'))
        self.critic1.load_state_dict(
            torch.load(
                path + "critic1.pth",
                map_location = 'cpu'))
        self.critic2.load_state_dict(
            torch.load(
                path + "critic2.pth",
                map_location = 'cpu'))
        self.transformer\
            .load_state_dict(
            torch.load(
                path +
                "transformer.pth",
                map_location = 'cpu'))
        print(f"Loaded from {path}")
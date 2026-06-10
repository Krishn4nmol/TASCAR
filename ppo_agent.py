# ppo_agent.py
# This is the brain of the RL system
# Implements the PPO (Proximal Policy Optimization) algorithm
# Contains the neural network that learns to scale queues
# From Section 4.2 of the paper

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from config import *

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────
# ACTOR NETWORK
# Decides which action to take
# ─────────────────────────────────────────

class ActorNetwork(nn.Module):
    """
    The Actor network.
    Takes current state as input.
    Outputs probability of each possible action.
    
    Think of it as the decision maker:
    Given what I see, what should I do?
    
    Architecture from paper Section 4.2:
    Input(state_dim) -> Hidden(128) -> Output(action_dim)
    """

    def __init__(self, state_dim, action_dim,
                 hidden_size=HIDDEN_LAYER_SIZE):
        super(ActorNetwork, self).__init__()

        # Neural network layers
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(state_dim, hidden_size),
            # Tanh activation (paper recommends this)
            nn.Tanh(),
            # Hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # Output layer - one value per possible action
            nn.Linear(hidden_size, action_dim),
            # Softmax converts to probabilities
            # Each action gets a probability 0 to 1
            # All probabilities sum to 1
            nn.Softmax(dim=-1)
        )

        # Orthogonal initialization
        # Makes training more stable
        # Recommended in paper reference [39]
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights orthogonally for stability"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        """
        Forward pass through network.
        Input:  state tensor
        Output: action probabilities
        """
        return self.network(state)


# ─────────────────────────────────────────
# CRITIC NETWORK
# Evaluates how good the current state is
# ─────────────────────────────────────────

class CriticNetwork(nn.Module):
    """
    The Critic network.
    Takes current state as input.
    Outputs a single value estimating
    how good this state is (value function).
    
    Think of it as the evaluator:
    How well am I doing right now?
    
    Same architecture as Actor but
    output is single value not probabilities.
    """

    def __init__(self, state_dim,
                 hidden_size=HIDDEN_LAYER_SIZE):
        super(CriticNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # Single output: estimated state value
            nn.Linear(hidden_size, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights orthogonally"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        """
        Forward pass through network.
        Input:  state tensor
        Output: single value estimate
        """
        return self.network(state)


# ─────────────────────────────────────────
# REPLAY BUFFER
# Stores experiences for training
# ─────────────────────────────────────────

class ReplayBuffer:
    """
    Stores (state, action, reward, next_state) tuples.
    Agent collects experiences then learns from them.
    
    PPO is on-policy so buffer is cleared after each update.
    Buffer size from paper Table 2: 1000.
    """

    def __init__(self, buffer_size=REPLAY_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.clear()

    def clear(self):
        """Clear all stored experiences"""
        self.states      = []
        self.actions     = []
        self.log_probs   = []
        self.rewards     = []
        self.next_states = []
        self.dones       = []
        self.count       = 0

    def store(self, state, action, log_prob,
              reward, next_state, done):
        """
        Store one experience tuple.
        Called after each environment step.
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.count += 1

    def get_all(self):
        """
        Returns all stored experiences as tensors.
        Called when it is time to update the network.
        """
        states      = torch.FloatTensor(
            np.array(self.states))
        actions     = torch.LongTensor(
            np.array(self.actions))
        log_probs   = torch.FloatTensor(
            np.array(self.log_probs))
        rewards     = torch.FloatTensor(
            np.array(self.rewards))
        next_states = torch.FloatTensor(
            np.array(self.next_states))
        dones       = torch.FloatTensor(
            np.array(self.dones))

        return (states, actions, log_probs,
                rewards, next_states, dones)

    def is_ready(self):
        """Returns True when buffer has enough experiences"""
        return self.count >= self.buffer_size


# ─────────────────────────────────────────
# PPO AGENT
# The complete learning agent
# ─────────────────────────────────────────

class PPOAgent:
    """
    Complete PPO Agent.
    Combines Actor, Critic, and ReplayBuffer.
    
    Implements Algorithm 2 from the paper:
    - Collects experiences using current policy
    - Computes advantages using GAE
    - Updates policy using PPO-clip objective
    - Adds entropy bonus for exploration
    
    From Section 4.2 of paper.
    """

    def __init__(self, state_dim, action_dim):

        self.state_dim  = state_dim
        self.action_dim = action_dim

        # Create actor and critic networks
        self.actor  = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)

        # Separate optimizers for actor and critic
        # Paper Table 2: learning rate = 0.001
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=LEARNING_RATE_ACTOR)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=LEARNING_RATE_CRITIC)

        # Replay buffer to store experiences
        self.buffer = ReplayBuffer()

        # Training hyperparameters from paper Table 2
        self.gamma      = DISCOUNT_FACTOR    # 0.63
        self.gae_lambda = GAE_LAMBDA         # 0.95
        self.clip_eps   = PPO_CLIP           # 0.2
        self.entropy    = ENTROPY_COEFF      # 0.01
        self.epochs     = EPOCHS_PER_UPDATE  # 10
        self.batch_size = MINI_BATCH_SIZE    # 20

        # Track training statistics
        self.training_step = 0
        self.actor_losses  = []
        self.critic_losses = []

    def choose_action(self, state):
        """
        Chooses action given current state.
        Returns action index and its log probability.
        
        Log probability is needed for PPO update.
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get action probabilities from actor network
        with torch.no_grad():
            action_probs = self.actor(state_tensor)

        # Create probability distribution
        dist = Categorical(action_probs)

        # Sample action from distribution
        # This provides exploration naturally
        action = dist.sample()

        # Get log probability of chosen action
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def store_experience(self, state, action,
                         log_prob, reward,
                         next_state, done):
        """Store one experience in replay buffer"""
        self.buffer.store(
            state, action, log_prob,
            reward, next_state, done)

    def update(self):
        """
        Updates actor and critic networks.
        Called when replay buffer is full.
        
        Implements the PPO-clip update from
        Algorithm 2 and Equations 7-12 of paper.
        """
        if not self.buffer.is_ready():
            return None, None

        # Get all experiences from buffer
        (states, actions, old_log_probs,
         rewards, next_states, dones) = (
            self.buffer.get_all())

        # Calculate advantages using GAE
        # GAE helps reduce variance in training
        advantages, returns = self._compute_gae(
            states, rewards, next_states, dones)

        # Normalize advantages
        # Makes training more stable
        advantages = ((advantages - advantages.mean()) /
                     (advantages.std() + 1e-8))

        # Track losses for logging
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        update_count      = 0

        # Multiple epochs of updates
        # Paper Table 2: 10 epochs
        for epoch in range(self.epochs):

            # Shuffle experiences for better learning
            indices = torch.randperm(len(states))

            # Mini-batch updates
            for start in range(0, len(states),
                              self.batch_size):
                end   = start + self.batch_size
                batch = indices[start:end]

                # Get batch data
                batch_states     = states[batch]
                batch_actions    = actions[batch]
                batch_old_probs  = old_log_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns    = returns[batch]

                # Get new action probabilities
                new_probs  = self.actor(batch_states)
                dist       = Categorical(new_probs)
                new_log_probs = dist.log_prob(batch_actions)

                # Calculate probability ratio
                # r_t(theta) from Equation 9 of paper
                ratio = torch.exp(
                    new_log_probs - batch_old_probs)

                # PPO-clip objective
                # Equation 8 of paper
                surr1 = ratio * batch_advantages
                surr2 = (torch.clamp(
                    ratio,
                    1 - self.clip_eps,
                    1 + self.clip_eps) * batch_advantages)

                # Take minimum to prevent large updates
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus for exploration
                # Equation 11 of paper
                entropy_bonus = dist.entropy().mean()

                # Combined actor loss
                # Equation 12 of paper
                actor_loss = (policy_loss -
                             self.entropy * entropy_bonus)

                # Critic loss: MSE between predicted
                # and actual returns
                values      = self.critic(
                    batch_states).squeeze()
                critic_loss = nn.MSELoss()(
                    values, batch_returns)

                # Update actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # Update critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                update_count      += 1

        # Clear buffer after update
        self.buffer.clear()
        self.training_step += 1

        avg_actor_loss  = total_actor_loss / max(1,
                                                update_count)
        avg_critic_loss = total_critic_loss / max(1,
                                                 update_count)

        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)

        return avg_actor_loss, avg_critic_loss

    def _compute_gae(self, states, rewards,
                     next_states, dones):
        """
        Computes Generalized Advantage Estimation.
        GAE reduces variance while keeping bias low.
        Makes training more stable and efficient.
        
        From paper Section 4.2 and Algorithm 2.
        """
        with torch.no_grad():
            values      = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()

        advantages = torch.zeros_like(rewards)
        gae        = 0.0

        # Calculate advantages in reverse order
        for t in reversed(range(len(rewards))):
            if dones[t]:
                # Episode ended - reset GAE
                delta = (rewards[t] -
                        values[t].item())
                gae   = delta
            else:
                delta = (rewards[t] +
                        self.gamma *
                        next_values[t].item() -
                        values[t].item())
                gae   = (delta +
                        self.gamma *
                        self.gae_lambda * gae)

            advantages[t] = gae

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def save(self, path):
        """Save trained model to disk"""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),
                   path + "actor.pth")
        torch.save(self.critic.state_dict(),
                   path + "critic.pth")
        print(f"Model saved to {path}")

    def load(self, path):
        """Load trained model from disk"""
        self.actor.load_state_dict(
            torch.load(
                path + "actor.pth",
                map_location = 'cpu'))
        self.critic.load_state_dict(
            torch.load(
                path + "critic.pth",
                map_location = 'cpu'))
        print(f"Model loaded from {path}")


# ─────────────────────────────────────────
# TEST THIS FILE
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Testing ppo_agent.py")
    print("=" * 50)

    # Environment dimensions
    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES

    print(f"\nCreating PPO Agent:")
    print(f"  State dimension:  {state_dim}")
    print(f"  Action dimension: {action_dim}")

    # Create agent
    agent = PPOAgent(state_dim, action_dim)

    print(f"\nActor Network:")
    print(f"  {agent.actor}")

    print(f"\nCritic Network:")
    print(f"  {agent.critic}")

    # Test action selection
    print(f"\nTesting action selection...")
    dummy_state  = np.random.randn(state_dim)
    action, prob = agent.choose_action(dummy_state)
    print(f"  State shape: {dummy_state.shape}")
    print(f"  Action chosen: {action}")
    print(f"  Log probability: {prob:.4f}")

    # Test storing experiences
    print(f"\nTesting experience storage...")
    for i in range(REPLAY_BUFFER_SIZE):
        state      = np.random.randn(state_dim)
        action_idx = np.random.randint(0, action_dim)
        log_prob   = np.random.randn()
        reward     = np.random.randn()
        next_state = np.random.randn(state_dim)
        done       = False

        agent.store_experience(
            state, action_idx, log_prob,
            reward, next_state, done)

    print(f"  Stored {agent.buffer.count} experiences")

    # Test network update
    print(f"\nTesting network update...")
    a_loss, c_loss = agent.update()
    print(f"  Actor loss:  {a_loss:.4f}")
    print(f"  Critic loss: {c_loss:.4f}")

    # Test save
    print(f"\nTesting model save...")
    agent.save(MODEL_SAVE_PATH)

    print("\n✅ ppo_agent.py working correctly!")
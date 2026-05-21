# environment.py
# This file creates the RL environment
# It connects the PPO agent to S-Cache
# The agent observes state, takes action, gets reward
# This implements Section 4.2 of the paper

import numpy as np
from scache import SCache
from simulator import AzureDataLoader
from config import *

class ServerlessEnvironment:
    """
    RL Environment for S-Cache queue scaling.
    
    Think of it like a game:
    - The agent is the player
    - The environment is the game world
    - State = what the agent sees
    - Action = what the agent does
    - Reward = score the agent gets
    
    Implements the MDP (Markov Decision Process)
    from Section 4.2 of the paper.
    """

    def __init__(self, function_calls, theta=THETA):

        # All function calls for this episode
        self.function_calls = function_calls

        # Theta: balance between cold starts and memory
        # 0.8 = slightly prioritize cold start reduction
        self.theta = theta

        # Current position in function calls list
        self.call_index = 0

        # The S-Cache being controlled by agent
        self.scache = SCache()

        # Step counter
        self.step_count = 0

        # Statistics tracked per step
        self.step_cold_starts = 0
        self.step_warm_starts = 0
        self.step_wmt_before  = 0.0
        self.step_wmt_after   = 0.0

        # For reward normalization
        # These track min/max values seen during training
        self.r1_min =  float('inf')
        self.r1_max = -float('inf')
        self.r2_min =  float('inf')
        self.r2_max = -float('inf')

        # State dimension per queue = 7 metrics
        # Total state size = NUM_QUEUES * 7
        self.state_dim  = NUM_QUEUES * 7

        # Action space: for each queue (-1, 0, +1)
        # -1 = shrink, 0 = keep same, +1 = expand
        # Total actions = 3^NUM_QUEUES combinations
        self.action_dim = 3 ** NUM_QUEUES

        # Map action index to per-queue decisions
        self.action_map = self._build_action_map()

    def _build_action_map(self):
        """
        Builds mapping from action index to
        per-queue scaling decisions.
        
        Example with 3 queues:
        Action 0 = [-1, -1, -1] shrink all
        Action 13 = [0, 0, 0] do nothing
        Action 26 = [+1, +1, +1] expand all
        
        Each value maps to scaling factor:
        -1 → -SCALING_FACTOR (shrink by 25%)
         0 → 0 (no change)
        +1 → +SCALING_FACTOR (expand by 25%)
        """
        action_map = {}
        choices    = [-SCALING_FACTOR, 0, SCALING_FACTOR]

        # Generate all combinations
        # For 3 queues: 3^3 = 27 possible actions
        for i in range(3 ** NUM_QUEUES):
            action = []
            temp   = i
            for _ in range(NUM_QUEUES):
                action.append(choices[temp % 3])
                temp //= 3
            action_map[i] = action

        return action_map

    def reset(self):
        """
        Resets environment for new episode.
        Called at start of each training episode.
        Returns initial state.
        """
        # Reset S-Cache
        self.scache = SCache()

        # Reset position in function calls
        self.call_index = 0

        # Reset step counter
        self.step_count = 0

        # Reset statistics
        self.step_cold_starts = 0
        self.step_warm_starts = 0
        self.step_wmt_before  = 0.0
        self.step_wmt_after   = 0.0

        # Reset per-step counters in S-Cache
        self.scache.reset_invocation_counters()

        # Return initial state
        return self._get_state()

    def step(self, action_idx):
        """
        Main step function.
        Agent takes an action, environment responds.
        
        This is called every DELTA function invocations.
        
        Returns:
        - next_state: what agent sees after action
        - reward: how good the action was
        - done: is episode finished
        """
        self.step_count += 1

        # Record WMT before action
        self.step_wmt_before = (
            self.scache.get_total_wasted_memory_time())

        # Reset step counters
        self.step_cold_starts = 0
        self.step_warm_starts = 0
        self.scache.reset_invocation_counters()

        # Apply the action to S-Cache queues
        scaling_decisions = self.action_map[action_idx]
        for q_idx, scale in enumerate(scaling_decisions):
            if scale != 0:
                self.scache.scale_queue(q_idx, scale)

        # Process DELTA function calls
        calls_processed = 0
        while (self.call_index < len(self.function_calls)
               and calls_processed < DELTA):

            call    = self.function_calls[self.call_index]
            is_warm = self.scache.handle_request(call)

            if is_warm:
                self.step_warm_starts += 1
            else:
                self.step_cold_starts += 1

            self.call_index    += 1
            calls_processed    += 1

        # Record WMT after processing
        self.step_wmt_after = (
            self.scache.get_total_wasted_memory_time())

        # Calculate reward
        reward = self._calculate_reward()

        # Get next state
        next_state = self._get_state()

        # Check if episode is done
        done = (self.call_index >= 
                len(self.function_calls))

        return next_state, reward, done

    def _get_state(self):
        """
        Gets current state observation.
        
        State = metrics for each queue:
        [cap, len, invocations, cold_starts,
         evictions, running, wmt]
        
        State is normalized to mean=0, std=1
        for better neural network training.
        
        From paper Section 4.2 State Space.
        """
        raw_state = self.scache.get_state()
        state     = np.array(raw_state, dtype=np.float32)

        # Normalize state
        # Prevents large numbers from confusing neural network
        mean = np.mean(state)
        std  = np.std(state)

        if std > 0:
            state = (state - mean) / std
        else:
            state = state - mean

        return state

    def _calculate_reward(self):
        """
        Calculates reward for the agent.
        
        R1 = cold starts this step (want to minimize)
        R2 = change in wasted memory time (want to minimize)
        
        Combined reward balances both objectives using theta.
        Reward is negative because we minimize both.
        
        From paper Section 4.2 Reward and Equation 6.
        """
        # R1: Total cold starts across all queues
        r1 = float(self.step_cold_starts)

        # R2: Change in wasted memory time
        r2 = float(self.step_wmt_after - 
                  self.step_wmt_before)
        r2 = max(0, r2)  # Only penalize increases

        # Update normalization bounds
        if r1 < self.r1_min: self.r1_min = r1
        if r1 > self.r1_max: self.r1_max = r1
        if r2 < self.r2_min: self.r2_min = r2
        if r2 > self.r2_max: self.r2_max = r2

        # Normalize R1
        r1_range = self.r1_max - self.r1_min
        if r1_range > 0:
            r1_norm = (r1 - self.r1_min) / r1_range
        else:
            r1_norm = 0.0

        # Normalize R2
        r2_range = self.r2_max - self.r2_min
        if r2_range > 0:
            r2_norm = (r2 - self.r2_min) / r2_range
        else:
            r2_norm = 0.0

        # Combined reward (negative = minimize both)
        # Equation 6 from paper
        reward = -(self.theta * r1_norm +
                  (1 - self.theta) * r2_norm)

        return float(reward)

    def get_current_metrics(self):
        """
        Returns current performance metrics.
        Used for logging during training.
        """
        total = (self.step_cold_starts + 
                self.step_warm_starts)

        cold_rate = (self.step_cold_starts / total * 100
                    if total > 0 else 0)

        return {
            'cold_start_rate': cold_rate,
            'cold_starts':     self.step_cold_starts,
            'warm_starts':     self.step_warm_starts,
            'wmt':             self.step_wmt_after,
            'queue_capacities': [
                q.capacity
                for q in self.scache.queues]
        }


# ─────────────────────────────────────────
# TEST THIS FILE
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Testing environment.py")
    print("=" * 50)

    # Load data
    loader = AzureDataLoader()
    calls  = loader.load_day(1)

    # Use small sample
    test_calls = calls[:50000]

    # Create environment
    env = ServerlessEnvironment(test_calls)

    print(f"\nEnvironment created:")
    print(f"  State dimension:  {env.state_dim}")
    print(f"  Action dimension: {env.action_dim}")
    print(f"  Total calls:      {len(test_calls)}")

    # Test reset
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")

    # Test a few random steps
    print(f"\nTesting 3 random steps...")
    total_reward = 0.0

    for i in range(3):
        # Random action
        action    = np.random.randint(0, env.action_dim)
        next_state, reward, done = env.step(action)

        metrics = env.get_current_metrics()
        total_reward += reward

        print(f"\n  Step {i+1}:")
        print(f"    Action:         {action}")
        print(f"    Reward:         {reward:.4f}")
        print(f"    Cold Start Rate:{metrics['cold_start_rate']:.1f}%")
        print(f"    Queue Caps:     {metrics['queue_capacities']}")
        print(f"    Done:           {done}")

        if done:
            break

    print(f"\nTotal reward: {total_reward:.4f}")
    print("\n✅ environment.py working correctly!")
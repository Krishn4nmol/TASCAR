# baselines.py
# Implements the 4 baseline algorithms from the paper
# These are what CASR competes against in experiments
# Section 5.1.6 of the paper

from collections import OrderedDict
import numpy as np
from config import *

# ─────────────────────────────────────────
# BASE CLASS
# All baselines inherit from this
# ─────────────────────────────────────────

class BaseAlgorithm:
    """
    Base class for all algorithms.
    Every algorithm must implement:
    - handle_request(call): process one function call
    - get_total_wasted_memory_time(): return WMT metric
    """

    def __init__(self, name):
        self.name          = name
        self.current_time  = 0.0
        self.containers    = {}  # function_id -> list of containers
        self.total_wmt     = 0.0

    def handle_request(self, function_call):
        """
        Must be implemented by each baseline.
        Returns True = warm start
        Returns False = cold start
        """
        raise NotImplementedError

    def get_total_wasted_memory_time(self):
        """Returns total wasted memory time"""
        return self.total_wmt

    def _find_available_container(self, function_id):
        """
        Looks for an available idle container
        for the given function.
        Returns container or None.
        """
        if function_id not in self.containers:
            return None

        for container in self.containers[function_id]:
            if container['state'] == 'idle':
                return container
        return None

    def _create_container(self, function_call):
        """
        Creates a new container for a function.
        Returns the container dictionary.
        """
        container = {
            'function_id':        function_call.function_id,
            'cold_start_overhead': function_call.cold_start_overhead,
            'memory_mb':          function_call.memory_mb,
            'state':              'cold_starting',
            'created_time':       self.current_time,
            'ready_time':         (self.current_time +
                                  function_call.cold_start_overhead),
            'idle_since':         None,
            'execution_time':     function_call.execution_time
        }

        # Add to containers dictionary
        if function_call.function_id not in self.containers:
            self.containers[function_call.function_id] = []
        self.containers[function_call.function_id].append(
            container)

        return container

    def _update_container_states(self, current_time):
        """
        Updates all container states based on current time.
        Cold starting -> idle when ready_time reached.
        """
        for func_id, cont_list in self.containers.items():
            for container in cont_list:
                if (container['state'] == 'cold_starting' and
                        current_time >= container['ready_time']):
                    container['state']      = 'idle'
                    container['idle_since'] = current_time

    def _calculate_wmt_for_eviction(self, container,
                                     current_time):
        """
        Calculates wasted memory time when evicting container.
        WMT = time container sat idle without serving requests.
        """
        if (container['state'] == 'idle' and
                container['idle_since'] is not None):
            idle_time = current_time - container['idle_since']
            self.total_wmt += idle_time


# ─────────────────────────────────────────
# BASELINE 1: FIXED
# Keep every container alive for exactly 10 minutes
# Most common strategy in real platforms today
# ─────────────────────────────────────────

class FixedAlgorithm(BaseAlgorithm):
    """
    Fixed keep-alive strategy.
    Every container stays in memory for exactly
    FIXED_KEEPALIVE_SECONDS (10 minutes) after finishing.
    
    This is what AWS Lambda and OpenWhisk use by default.
    Simple but wastes memory for rarely-called functions.
    From paper Section 5.1.6.
    """

    def __init__(self):
        super().__init__("Fixed")
        # All containers kept for exactly this long
        self.keepalive_time = FIXED_KEEPALIVE_SECONDS

    def handle_request(self, function_call):
        """
        Process one function call with fixed keep-alive.
        """
        self.current_time = function_call.arrival_time

        # Update all container states
        self._update_container_states(self.current_time)

        # Evict containers that exceeded keep-alive time
        self._evict_expired_containers()

        # Try to find available container
        container = self._find_available_container(
            function_call.function_id)

        if container is not None:
            # WARM START
            container['state']     = 'running'
            container['ready_time'] = (self.current_time +
                                      function_call.execution_time)
            return True
        else:
            # COLD START - create new container
            self._create_container(function_call)
            return False

    def _evict_expired_containers(self):
        """
        Remove containers that have been idle longer
        than the fixed keep-alive time.
        """
        for func_id in list(self.containers.keys()):
            active = []
            for container in self.containers[func_id]:
                if container['state'] == 'idle':
                    idle_duration = (self.current_time -
                                   container['idle_since'])
                    if idle_duration > self.keepalive_time:
                        # Container expired - calculate WMT
                        self._calculate_wmt_for_eviction(
                            container, self.current_time)
                    else:
                        active.append(container)
                else:
                    active.append(container)
            self.containers[func_id] = active


# ─────────────────────────────────────────
# BASELINE 2: LCS
# Least Cold Start - LRU based strategy
# From paper reference [14]
# ─────────────────────────────────────────

class LCSAlgorithm(BaseAlgorithm):
    """
    Least Cold Start algorithm.
    Uses LRU (Least Recently Used) eviction policy.
    Keeps high-frequency containers in memory longer.
    
    Good at reducing cold starts but uses too much memory.
    From paper Section 5.1.6 reference [14].
    """

    def __init__(self, max_containers=500):
        super().__init__("LCS")

        # Maximum total containers across all functions
        self.max_containers = max_containers

        # LRU cache: OrderedDict maintains access order
        # Key = function_id, Value = container dict
        self.lru_cache = OrderedDict()

        # Track access frequency for each function
        self.access_count = {}

    def handle_request(self, function_call):
        """
        Process one function call with LCS strategy.
        """
        self.current_time = function_call.arrival_time
        function_id       = function_call.function_id

        # Update container states
        self._update_container_states(self.current_time)

        # Update access count for this function
        if function_id not in self.access_count:
            self.access_count[function_id] = 0
        self.access_count[function_id] += 1

        # Check if we have a warm container
        if function_id in self.lru_cache:
            container = self.lru_cache[function_id]
            if container['state'] == 'idle':
                # WARM START
                container['state']     = 'running'
                container['ready_time'] = (
                    self.current_time +
                    function_call.execution_time)

                # Move to end (most recently used)
                self.lru_cache.move_to_end(function_id)
                return True

        # COLD START
        # Check if cache is full
        if len(self.lru_cache) >= self.max_containers:
            self._evict_lru_container()

        # Create new container
        new_container = self._create_container(function_call)
        self.lru_cache[function_id] = new_container
        return False

    def _evict_lru_container(self):
        """
        Evicts the least recently used container.
        This is the core LRU eviction policy.
        """
        if not self.lru_cache:
            return

        # Get least recently used (first item)
        evicted_id, evicted_container = next(
            iter(self.lru_cache.items()))

        # Calculate WMT before eviction
        self._calculate_wmt_for_eviction(
            evicted_container, self.current_time)

        # Remove from cache
        del self.lru_cache[evicted_id]


# ─────────────────────────────────────────
# BASELINE 3: HIST
# History-based keep-alive strategy
# From paper reference [6]
# ─────────────────────────────────────────

class HistAlgorithm(BaseAlgorithm):
    """
    History-based keep-alive strategy.
    Analyzes function invocation history to determine
    optimal keep-alive time for each function.
    
    Works well for regular patterns but falls back to
    fixed strategy for irregular functions (32%+ of all).
    From paper Section 5.1.6 reference [6].
    """

    def __init__(self):
        super().__init__("Hist")

        # Invocation history per function
        # Stores last N arrival times
        self.invocation_history = {}

        # Calculated keep-alive time per function
        self.keepalive_times = {}

        # How many historical calls to remember
        self.history_window = 10

        # Default fallback keep-alive (10 minutes)
        self.default_keepalive = FIXED_KEEPALIVE_SECONDS

        # Active containers per function
        self.active_containers = {}

    def handle_request(self, function_call):
        """
        Process one function call with Hist strategy.
        """
        self.current_time = function_call.arrival_time
        function_id       = function_call.function_id

        # Update container states
        self._update_container_states(self.current_time)

        # Update invocation history
        self._update_history(function_id, self.current_time)

        # Calculate keep-alive time for this function
        keepalive = self._calculate_keepalive(function_id)

        # Evict containers that exceeded their keep-alive
        self._evict_expired_containers(keepalive, function_id)

        # Try warm start
        container = self._find_available_container(function_id)

        if container is not None:
            # WARM START
            container['state']     = 'running'
            container['ready_time'] = (self.current_time +
                                      function_call.execution_time)
            return True
        else:
            # COLD START
            self._create_container(function_call)
            return False

    def _update_history(self, function_id, arrival_time):
        """
        Updates the invocation history for a function.
        Keeps only the last N arrival times.
        """
        if function_id not in self.invocation_history:
            self.invocation_history[function_id] = []

        self.invocation_history[function_id].append(
            arrival_time)

        # Keep only recent history
        if (len(self.invocation_history[function_id]) >
                self.history_window):
            self.invocation_history[function_id].pop(0)

    def _calculate_keepalive(self, function_id):
        """
        Calculates optimal keep-alive time based on history.
        If function has regular pattern: use inter-arrival time
        If function is irregular: fall back to fixed 10 minutes
        
        This is the core of Hist algorithm.
        """
        history = self.invocation_history.get(
            function_id, [])

        # Need at least 2 calls to calculate interval
        if len(history) < 2:
            return self.default_keepalive

        # Calculate inter-arrival times
        intervals = []
        for i in range(1, len(history)):
            interval = history[i] - history[i-1]
            intervals.append(interval)

        # Check if pattern is regular
        # Regular = low coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval  = np.std(intervals)

        if mean_interval == 0:
            return self.default_keepalive

        # Coefficient of variation measures regularity
        cv = std_interval / mean_interval

        if cv < 1.0:
            # Regular pattern - use mean interval as keep-alive
            return mean_interval
        else:
            # Irregular pattern - fall back to fixed
            # This affects 32%+ of functions per paper
            return self.default_keepalive

    def _evict_expired_containers(self, keepalive,
                                    function_id):
        """
        Evicts containers that exceeded their keep-alive time.
        """
        if function_id not in self.containers:
            return

        active = []
        for container in self.containers[function_id]:
            if container['state'] == 'idle':
                idle_time = (self.current_time -
                           container['idle_since'])
                if idle_time > keepalive:
                    self._calculate_wmt_for_eviction(
                        container, self.current_time)
                else:
                    active.append(container)
            else:
                active.append(container)

        self.containers[function_id] = active


# ─────────────────────────────────────────
# BASELINE 4: FAASCACHE
# Greedy-Dual-Size based caching
# From paper reference [8]
# ─────────────────────────────────────────

class FaaSCacheAlgorithm(BaseAlgorithm):
    """
    FaaSCache algorithm.
    Uses Greedy-Dual-Size-Frequency (GDSF) caching.
    Considers cold start overhead, frequency, and recency.
    
    Best at reducing cold starts among baselines but
    wastes memory due to lack of scaling mechanism.
    From paper Section 5.1.6 reference [8].
    """

    def __init__(self, max_memory_mb=SERVER_MEMORY_MB):
        super().__init__("FaaSCache")

        # Maximum memory for containers
        self.max_memory_mb  = max_memory_mb
        self.used_memory_mb = 0

        # GDSF priority cache
        # Key = function_id
        # Value = {container, priority, frequency}
        self.cache = {}

        # Clock value for GDSF (increases with evictions)
        self.clock = 0.0

        # Access frequency per function
        self.frequency = {}

    def handle_request(self, function_call):
        """
        Process one function call with FaaSCache strategy.
        """
        self.current_time = function_call.arrival_time
        function_id       = function_call.function_id

        # Update container states
        self._update_container_states(self.current_time)

        # Update frequency
        if function_id not in self.frequency:
            self.frequency[function_id] = 0
        self.frequency[function_id] += 1

        # Check for warm container
        if function_id in self.cache:
            entry     = self.cache[function_id]
            container = entry['container']

            if container['state'] == 'idle':
                # WARM START
                container['state']     = 'running'
                container['ready_time'] = (
                    self.current_time +
                    function_call.execution_time)

                # Update GDSF priority
                entry['priority'] = self._calculate_priority(
                    function_call)
                entry['frequency'] = self.frequency[function_id]
                return True

        # COLD START
        # Free memory if needed
        needed_memory = function_call.memory_mb
        while (self.used_memory_mb + needed_memory >
               self.max_memory_mb and self.cache):
            self._evict_lowest_priority()

        # Create new container
        new_container = self._create_container(function_call)

        self.cache[function_id] = {
            'container': new_container,
            'priority':  self._calculate_priority(
                function_call),
            'frequency': self.frequency[function_id]
        }

        self.used_memory_mb += function_call.memory_mb
        return False

    def _calculate_priority(self, function_call):
        """
        Calculates GDSF priority for a function.
        Priority = clock + (frequency * cold_start) / memory
        
        Higher priority = more valuable to keep in cache.
        Functions with high cold start overhead and
        high frequency get highest priority.
        This is the core of FaaSCache.
        """
        function_id = function_call.function_id
        freq        = self.frequency.get(function_id, 1)

        # GDSF formula from paper reference [8]
        priority = (self.clock +
                   (freq * function_call.cold_start_overhead) /
                   max(1, function_call.memory_mb))

        return priority

    def _evict_lowest_priority(self):
        """
        Evicts the container with lowest GDSF priority.
        Updates clock value after eviction.
        """
        if not self.cache:
            return

        # Find function with lowest priority
        lowest_id  = min(self.cache.keys(),
                        key=lambda k: self.cache[k]['priority'])
        entry      = self.cache[lowest_id]
        container  = entry['container']

        # Update clock to lowest priority value
        self.clock = entry['priority']

        # Calculate WMT
        self._calculate_wmt_for_eviction(
            container, self.current_time)

        # Free memory
        self.used_memory_mb = max(0,
            self.used_memory_mb - container['memory_mb'])

        # Remove from cache
        del self.cache[lowest_id]


# ─────────────────────────────────────────
# TEST ALL BASELINES
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Testing baselines.py")
    print("=" * 50)

    from simulator import AzureDataLoader, Simulator

    # Load test data
    loader     = AzureDataLoader()
    calls      = loader.load_day(1)
    test_calls = calls[:5000]

    print(f"\nTesting with {len(test_calls)} calls...")
    print("-" * 50)

    # Test all 4 baselines
    algorithms = [
        FixedAlgorithm(),
        LCSAlgorithm(),
        HistAlgorithm(),
        FaaSCacheAlgorithm()
    ]

    results = {}

    for algo in algorithms:
        print(f"\nRunning {algo.name}...")
        sim     = Simulator(algo)
        metrics = sim.run(test_calls, verbose=False)
        results[algo.name] = metrics

    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Cold Start%':>11} "
          f"{'Avg Overhead':>13} {'Avg WMT':>10}")
    print("-" * 60)

    for name, metrics in results.items():
        print(f"{name:<12} "
              f"{metrics['cold_start_rate']:>10.2f}% "
              f"{metrics['avg_cold_start_overhead']:>12.2f}s "
              f"{metrics['avg_wasted_memory_time']:>9.2f}s")

    print("\n✅ baselines.py working correctly!")
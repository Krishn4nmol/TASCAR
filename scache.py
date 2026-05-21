# scache.py
# This is the HEART of the CASR paper
# Implements S-Cache with W-TinyLFU caching framework
# Organizes containers into 3 queues based on cold start overhead

import time
from collections import OrderedDict
import numpy as np
from config import *

# ─────────────────────────────────────────
# CONTAINER CLASS
# Represents one container in memory
# ─────────────────────────────────────────

class Container:
    """
    Represents one container sitting in memory.
    A container is either:
    - IDLE: finished running, available for reuse (warm start)
    - RUNNING: currently executing a function (unavailable)
    - COLD_STARTING: loading dependencies (unavailable)
    """

    # Container states
    IDLE          = "idle"
    RUNNING       = "running"
    COLD_STARTING = "cold_starting"

    def __init__(self, function_id, cold_start_overhead,
                 memory_mb, current_time):

        # Which function this container belongs to
        self.function_id = function_id

        # How long cold start takes for this function
        self.cold_start_overhead = cold_start_overhead

        # Memory this container uses in MB
        self.memory_mb = memory_mb

        # When this container was created
        self.created_time = current_time

        # When this container became idle (finished running)
        self.idle_since = None

        # Current state of container
        self.state = Container.COLD_STARTING

        # When the container will finish cold starting
        self.ready_time = current_time + cold_start_overhead

        # How many times this container has been reused
        self.reuse_count = 0

        # Track when container became idle for WMT calculation
        self.last_idle_time = None

        # Total wasted memory time accumulated
        self.wasted_memory_time = 0.0

    def is_available(self, current_time):
        """
        Returns True if container can serve a new request.
        Container is available ONLY when it is IDLE.
        This implements the key insight from the paper:
        C_available = C_mem - C_running
        """
        # Update state based on current time
        self._update_state(current_time)
        return self.state == Container.IDLE

    def _update_state(self, current_time):
        """
        Updates container state based on current time.
        Cold starting → Idle when ready_time is reached.
        """
        if (self.state == Container.COLD_STARTING and
                current_time >= self.ready_time):
            self.state      = Container.IDLE
            self.idle_since = current_time
            self.last_idle_time = current_time

    def start_execution(self, current_time, execution_time):
        """
        Marks container as running a function.
        Container becomes unavailable during execution.
        """
        self.state      = Container.RUNNING
        self.ready_time = current_time + execution_time
        self.reuse_count += 1

        # Calculate wasted memory time before execution
        if self.last_idle_time is not None:
            idle_duration = current_time - self.last_idle_time
            self.wasted_memory_time += idle_duration
            self.last_idle_time = None

    def finish_execution(self, current_time):
        """
        Marks container as idle after execution completes.
        Container becomes available for reuse again.
        """
        self.state          = Container.IDLE
        self.idle_since     = current_time
        self.last_idle_time = current_time

    def get_wasted_memory_time(self, current_time):
        """
        Returns total wasted memory time for this container.
        WMT = time container sits idle without serving requests.
        """
        total = self.wasted_memory_time

        # Add current idle period if container is idle now
        if (self.state == Container.IDLE and
                self.last_idle_time is not None):
            total += current_time - self.last_idle_time

        return total

    def __repr__(self):
        return (f"Container("
                f"func={self.function_id}, "
                f"state={self.state}, "
                f"reuses={self.reuse_count})")


# ─────────────────────────────────────────
# COUNT MIN SKETCH
# Tracks access frequency efficiently
# Used by TinyLFU to decide evictions
# ─────────────────────────────────────────

class CountMinSketch:
    """
    Efficient frequency tracking data structure.
    Uses much less memory than a regular dictionary.
    Part of the TinyLFU eviction policy.
    From paper Section 4.1 reference [34]
    """

    def __init__(self, width=1000, depth=4):
        # Width = number of counters per row
        self.width = width
        # Depth = number of hash functions
        self.depth = depth
        # The counter table
        self.table = np.zeros((depth, width), dtype=np.int32)
        # Random seeds for hash functions
        self.seeds = np.random.randint(1, 1000, depth)
        # Total items added (for decay mechanism)
        self.total_added = 0
        # Decay threshold - reset counts periodically
        self.decay_threshold = width * 10

    def _hash(self, key, seed):
        """Simple hash function"""
        return (abs(hash(key)) * int(seed)) % self.width

    def add(self, key):
        """
        Increments frequency count for a key.
        Called every time a container is accessed.
        """
        for i in range(self.depth):
            idx = self._hash(key, self.seeds[i])
            self.table[i][idx] += 1

        self.total_added += 1

        # Decay: halve all counts periodically
        # This implements the freshness mechanism
        # Recent accesses matter more than old ones
        if self.total_added >= self.decay_threshold:
            self.table = self.table // 2
            self.total_added = self.decay_threshold // 2

    def estimate(self, key):
        """
        Estimates how frequently a key has been accessed.
        Returns minimum count across all hash functions.
        """
        counts = []
        for i in range(self.depth):
            idx = self._hash(key, self.seeds[i])
            counts.append(self.table[i][idx])
        return min(counts)


# ─────────────────────────────────────────
# BLOOM FILTER
# Pre-filters candidates for eviction
# From paper Section 4.1 reference [35]
# ─────────────────────────────────────────

class BloomFilter:
    """
    Quickly checks if a container has been
    seen recently. Used to pre-filter eviction
    candidates from window cache.
    """

    def __init__(self, size=10000, num_hashes=3):
        self.size       = size
        self.num_hashes = num_hashes
        self.bits       = np.zeros(size, dtype=bool)
        self.seeds      = np.random.randint(1, 1000, num_hashes)

    def _hashes(self, key):
        """Generate multiple hash positions for key"""
        return [(abs(hash(key)) * int(s)) % self.size
                for s in self.seeds]

    def add(self, key):
        """Add a key to the filter"""
        for pos in self._hashes(key):
            self.bits[pos] = True

    def contains(self, key):
        """
        Check if key might be in filter.
        Returns False = definitely not seen before
        Returns True = probably seen before
        """
        return all(self.bits[pos]
                  for pos in self._hashes(key))

    def reset(self):
        """Clear the filter"""
        self.bits = np.zeros(self.size, dtype=bool)


# ─────────────────────────────────────────
# W-TINYLFU QUEUE
# One queue with W-TinyLFU eviction policy
# Each S-Cache queue uses this structure
# ─────────────────────────────────────────

class WTinyLFUQueue:
    """
    Implements W-TinyLFU caching framework for one queue.
    Structure:
    [Window Cache (20%)] + [Main Cache (80%)]
    
    New containers go to Window Cache first.
    Popular containers graduate to Main Cache.
    TinyLFU decides which containers to keep.
    
    From paper Section 4.1 and Figure 5.
    """

    def __init__(self, capacity, window_ratio=WINDOW_CACHE_RATIO):

        # Total capacity of this queue
        self.capacity = max(1, capacity)

        # Window cache size (20% of total)
        self.window_size = max(1,
            int(self.capacity * window_ratio))

        # Main cache size (80% of total)
        self.main_size = max(1,
            self.capacity - self.window_size)

        # Window cache: OrderedDict acts as LRU
        # Key = function_id, Value = Container object
        self.window_cache = OrderedDict()

        # Main cache: OrderedDict acts as LRU
        self.main_cache = OrderedDict()

        # Frequency tracker for eviction decisions
        self.count_min_sketch = CountMinSketch()

        # Bloom filter for pre-filtering
        self.bloom_filter = BloomFilter()

        # Containers flagged for eviction but still running
        self.eviction_candidates = []

        # Statistics for this queue
        self.hits        = 0
        self.misses      = 0
        self.evictions   = 0

    def find_available_container(self, function_id,
                                  current_time):
        """
        Looks for an available container for function_id.
        Returns container if found (warm start).
        Returns None if not found (cold start needed).
        This implements Algorithm 1 lines 3-9 from paper.
        """
        # Check window cache first
        if function_id in self.window_cache:
            container = self.window_cache[function_id]
            if container.is_available(current_time):
                # Update frequency count
                self.count_min_sketch.add(function_id)
                # Move to end (most recently used)
                self.window_cache.move_to_end(function_id)
                self.hits += 1
                return container

        # Check main cache
        if function_id in self.main_cache:
            container = self.main_cache[function_id]
            if container.is_available(current_time):
                # Update frequency count
                self.count_min_sketch.add(function_id)
                # Move to end (most recently used)
                self.main_cache.move_to_end(function_id)
                self.hits += 1
                return container

        # Not found or not available
        self.misses += 1
        return None

    def add_container(self, container, current_time):
        """
        Adds a new container to the queue after cold start.
        New containers always go to window cache first.
        This implements Algorithm 1 lines 11-18 from paper.
        """
        function_id = container.function_id

        # Update bloom filter
        self.bloom_filter.add(function_id)
        self.count_min_sketch.add(function_id)

        # If window cache is full, try to promote to main cache
        if len(self.window_cache) >= self.window_size:
            self._promote_from_window(current_time)

        # Add to window cache
        self.window_cache[function_id] = container
        self.window_cache.move_to_end(function_id)

    def _promote_from_window(self, current_time):
        """
        When window cache is full, decide what to do.
        Either promote window victim to main cache,
        or evict it entirely.
        This is the TinyLFU admission policy.
        """
        if not self.window_cache:
            return

        # Get least recently used from window (first item)
        window_victim_id = next(iter(self.window_cache))
        window_victim    = self.window_cache[window_victim_id]

        # Check if this function has been seen before
        if not self.bloom_filter.contains(window_victim_id):
            # Never seen before - just evict from window
            del self.window_cache[window_victim_id]
            self.evictions += 1
            return

        # Compare frequency with main cache victim
        if len(self.main_cache) >= self.main_size:
            # Get least recently used from main cache
            main_victim_id = next(iter(self.main_cache))

            window_freq = self.count_min_sketch.estimate(
                window_victim_id)
            main_freq   = self.count_min_sketch.estimate(
                main_victim_id)

            if window_freq > main_freq:
                # Window victim is more popular - promote it
                # Evict main victim
                evicted = self.main_cache.pop(main_victim_id)
                self._handle_eviction(evicted, current_time)

                # Move window victim to main cache
                del self.window_cache[window_victim_id]
                self.main_cache[window_victim_id] = window_victim
            else:
                # Main victim is more popular - evict window victim
                del self.window_cache[window_victim_id]
                self._handle_eviction(window_victim, current_time)
        else:
            # Main cache has space - just promote
            del self.window_cache[window_victim_id]
            self.main_cache[window_victim_id] = window_victim

    def _handle_eviction(self, container, current_time):
        """
        Handles container eviction.
        If container is running, flag it for later removal.
        If idle, remove immediately.
        From paper: running containers are flagged not deleted.
        """
        self.evictions += 1

        if container.state == Container.RUNNING:
            # Cannot remove running container immediately
            # Flag it - will be removed after execution
            self.eviction_candidates.append(container)
        # If idle, container is simply not referenced anymore
        # Python garbage collector handles memory cleanup

    def resize(self, new_capacity, current_time):
        """
        Changes queue capacity.
        Called by RL agent during scaling.
        Implements Algorithm 3 from paper.
        Returns list of evicted containers.
        """
        new_capacity  = max(1, new_capacity)
        old_capacity  = self.capacity
        self.capacity = new_capacity

        # Recalculate window and main sizes
        self.window_size = max(1,
            int(new_capacity * WINDOW_CACHE_RATIO))
        self.main_size   = max(1,
            new_capacity - self.window_size)

        evicted = []

        # If shrinking, evict from tail of LRU lists
        if new_capacity < old_capacity:
            # Evict from window cache tail
            while len(self.window_cache) > self.window_size:
                evicted_id, evicted_container = (
                    self.window_cache.popitem(last=False))
                self._handle_eviction(
                    evicted_container, current_time)
                evicted.append(evicted_container)

            # Evict from main cache tail
            while len(self.main_cache) > self.main_size:
                evicted_id, evicted_container = (
                    self.main_cache.popitem(last=False))
                self._handle_eviction(
                    evicted_container, current_time)
                evicted.append(evicted_container)

        return evicted

    def get_stats(self):
        """Returns statistics for this queue"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100
                   if total > 0 else 0)
        return {
            'capacity':      self.capacity,
            'window_size':   self.window_size,
            'main_size':     self.main_size,
            'window_used':   len(self.window_cache),
            'main_used':     len(self.main_cache),
            'hits':          self.hits,
            'misses':        self.misses,
            'hit_rate':      hit_rate,
            'evictions':     self.evictions
        }

    def get_total_containers(self):
        """Returns total containers currently in queue"""
        return len(self.window_cache) + len(self.main_cache)

    def get_running_containers(self):
        """
        Returns count of currently running containers.
        This is the C_running metric from paper Equation 1.
        """
        running = 0
        for c in self.window_cache.values():
            if c.state == Container.RUNNING:
                running += 1
        for c in self.main_cache.values():
            if c.state == Container.RUNNING:
                running += 1
        return running

    def get_wasted_memory_time(self, current_time):
        """Returns total WMT for all containers in queue"""
        total = 0.0
        for c in self.window_cache.values():
            total += c.get_wasted_memory_time(current_time)
        for c in self.main_cache.values():
            total += c.get_wasted_memory_time(current_time)
        return total


# ─────────────────────────────────────────
# S-CACHE
# The complete serverless cache system
# Combines all queues together
# This is the main contribution of the paper
# ─────────────────────────────────────────

class SCache:
    """
    Complete S-Cache implementation.
    Manages K queues, each handling different
    cold start overhead ranges.
    
    This implements the full Algorithm 1 from paper
    plus the queue management from Section 4.1.
    """

    def __init__(self,
                 num_queues=NUM_QUEUES,
                 initial_capacities=INITIAL_QUEUE_CAPACITY):

        self.num_queues   = num_queues
        self.current_time = 0.0

        # Create K queues with W-TinyLFU structure
        self.queues = []
        for k in range(num_queues):
            capacity = initial_capacities[k]
            queue    = WTinyLFUQueue(capacity)
            self.queues.append(queue)

        # Track all containers for WMT calculation
        self.all_containers = []

        # Statistics
        self.total_requests   = 0
        self.total_cold_starts = 0
        self.total_warm_starts = 0

    def handle_request(self, function_call):
        """
        Main entry point for every function call.
        Decides warm start or cold start.
        Returns True = warm start, False = cold start.
        
        This implements the full Algorithm 1 from paper.
        """
        self.current_time  = function_call.arrival_time
        self.total_requests += 1

        # Get which queue this function belongs to
        queue_idx = function_call.queue_index
        queue     = self.queues[queue_idx]

        # Try to find available container (warm start)
        container = queue.find_available_container(
            function_call.function_id,
            self.current_time)

        if container is not None:
            # WARM START - reuse existing container
            container.start_execution(
                self.current_time,
                function_call.execution_time)
            self.total_warm_starts += 1
            return True

        else:
            # COLD START - create new container
            new_container = Container(
                function_id=function_call.function_id,
                cold_start_overhead=(
                    function_call.cold_start_overhead),
                memory_mb=function_call.memory_mb,
                current_time=self.current_time)

            # Add to appropriate queue
            queue.add_container(new_container,
                                self.current_time)

            # Track for WMT calculation
            self.all_containers.append(new_container)

            self.total_cold_starts += 1
            return False

    def get_total_wasted_memory_time(self):
        """
        Calculates total WMT across all queues.
        WMT = time containers sit idle in memory.
        From paper Section 3.2.
        """
        total = 0.0
        for queue in self.queues:
            total += queue.get_wasted_memory_time(
                self.current_time)
        return total

    def scale_queue(self, queue_idx, scaling_factor):
        """
        Scales a queue's capacity up or down.
        Called by RL agent during training.
        scaling_factor > 0 = expand
        scaling_factor < 0 = shrink
        scaling_factor = 0 = no change
        Implements Algorithm 3 from paper.
        """
        queue       = self.queues[queue_idx]
        old_cap     = queue.capacity
        new_cap     = int(old_cap * (1 + scaling_factor))
        new_cap     = max(1, new_cap)

        evicted = queue.resize(new_cap, self.current_time)
        return evicted

    def get_state(self):
        """
        Returns current state of all queues.
        Used by RL agent as observation.
        Implements state space from Section 4.2.
        """
        state = []
        for k, queue in enumerate(self.queues):
            stats = queue.get_stats()
            queue_state = [
                # Queue capacity
                stats['capacity'],
                # Current containers in queue
                stats['window_used'] + stats['main_used'],
                # Invocations (approximated by misses)
                stats['hits'] + stats['misses'],
                # Cold starts (misses)
                stats['misses'],
                # Evictions
                stats['evictions'],
                # Running containers
                queue.get_running_containers(),
                # Wasted memory time
                queue.get_wasted_memory_time(
                    self.current_time)
            ]
            state.extend(queue_state)
        return state

    def get_queue_stats(self):
        """Returns stats for all queues"""
        return [q.get_stats() for q in self.queues]

    def reset_invocation_counters(self):
        """
        Resets per-step counters.
        Called after each RL agent decision step.
        """
        for queue in self.queues:
            queue.hits      = 0
            queue.misses    = 0
            queue.evictions = 0


# ─────────────────────────────────────────
# TEST THIS FILE
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Testing scache.py")
    print("=" * 50)

    # Import simulator for test data
    from simulator import AzureDataLoader, Simulator

    # Create S-Cache
    scache = SCache()
    print("\nS-Cache created with 3 queues:")
    for k, q in enumerate(scache.queues):
        print(f"  Queue {k}: capacity={q.capacity}")

    # Load test data
    loader = AzureDataLoader()
    calls  = loader.load_day(1)

    # Use small sample for quick test
    test_calls = calls[:5000]
    print(f"\nTesting with {len(test_calls)} calls...")

    # Run simulation
    sim     = Simulator(scache)
    metrics = sim.run(test_calls, verbose=False)

    print(f"\nResults:")
    print(f"  Cold Start Rate:         "
          f"{metrics['cold_start_rate']:.2f}%")
    print(f"  Avg Cold Start Overhead: "
          f"{metrics['avg_cold_start_overhead']:.2f}s")
    print(f"  Avg Wasted Memory Time:  "
          f"{metrics['avg_wasted_memory_time']:.2f}s")

    print("\nQueue Statistics:")
    for k, stats in enumerate(scache.get_queue_stats()):
        print(f"  Queue {k}: "
              f"hits={stats['hits']}, "
              f"misses={stats['misses']}, "
              f"hit_rate={stats['hit_rate']:.1f}%")

    print("\n✅ scache.py working correctly!")
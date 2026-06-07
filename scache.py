# scache.py
# This is the HEART of the CASR paper
# Implements S-Cache with W-TinyLFU caching framework
# Organizes containers into 3 queues based on cold start overhead
# Extended with comprehensive metrics for TASCAR paper!

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

    def __init__(self, function_id,
                 cold_start_overhead,
                 memory_mb, current_time):

        self.function_id         = function_id
        self.cold_start_overhead = cold_start_overhead
        self.memory_mb           = memory_mb
        self.created_time        = current_time
        self.idle_since          = None
        self.state               = Container.COLD_STARTING
        self.ready_time          = (current_time +
                                    cold_start_overhead)
        self.reuse_count         = 0
        self.last_idle_time      = None
        self.wasted_memory_time  = 0.0

        # NEW: track active execution time
        self.total_active_time   = 0.0
        self.execution_start     = None

    def is_available(self, current_time):
        """
        Returns True if container can serve a new request.
        Container is available ONLY when it is IDLE.
        C_available = C_mem - C_running
        """
        self._update_state(current_time)
        return self.state == Container.IDLE

    def _update_state(self, current_time):
        if (self.state == Container.COLD_STARTING and
                current_time >= self.ready_time):
            self.state          = Container.IDLE
            self.idle_since     = current_time
            self.last_idle_time = current_time

    def start_execution(self, current_time,
                        execution_time):
        self.state          = Container.RUNNING
        self.ready_time     = current_time + execution_time
        self.reuse_count   += 1
        self.execution_start = current_time

        if self.last_idle_time is not None:
            idle_duration = (current_time -
                             self.last_idle_time)
            self.wasted_memory_time += idle_duration
            self.last_idle_time = None

    def finish_execution(self, current_time):
        self.state          = Container.IDLE
        self.idle_since     = current_time
        self.last_idle_time = current_time

        # Track active time
        if self.execution_start is not None:
            self.total_active_time += (
                current_time -
                self.execution_start)
            self.execution_start = None

    def get_wasted_memory_time(self, current_time):
        total = self.wasted_memory_time
        if (self.state == Container.IDLE and
                self.last_idle_time is not None):
            total += (current_time -
                      self.last_idle_time)
        return total

    def get_lifetime(self, current_time):
        """Total lifetime of container"""
        return current_time - self.created_time

    def get_utilization(self, current_time):
        """
        Container Utilization Rate
        CUR = active_time / total_lifetime
        """
        lifetime = self.get_lifetime(current_time)
        if lifetime <= 0:
            return 0.0
        return min(
            self.total_active_time /
            lifetime, 1.0)

    def __repr__(self):
        return (f"Container("
                f"func={self.function_id}, "
                f"state={self.state}, "
                f"reuses={self.reuse_count})")


# ─────────────────────────────────────────
# COUNT MIN SKETCH
# ─────────────────────────────────────────

class CountMinSketch:
    def __init__(self, width=1000, depth=4):
        self.width           = width
        self.depth           = depth
        self.table           = np.zeros(
            (depth, width), dtype=np.int32)
        self.seeds           = np.random.randint(
            1, 1000, depth)
        self.total_added     = 0
        self.decay_threshold = width * 10

    def _hash(self, key, seed):
        return (abs(hash(key)) *
                int(seed)) % self.width

    def add(self, key):
        for i in range(self.depth):
            idx = self._hash(
                key, self.seeds[i])
            self.table[i][idx] += 1
        self.total_added += 1
        if (self.total_added >=
                self.decay_threshold):
            self.table       = (
                self.table // 2)
            self.total_added = (
                self.decay_threshold // 2)

    def estimate(self, key):
        counts = []
        for i in range(self.depth):
            idx = self._hash(
                key, self.seeds[i])
            counts.append(
                self.table[i][idx])
        return min(counts)


# ─────────────────────────────────────────
# BLOOM FILTER
# ─────────────────────────────────────────

class BloomFilter:
    def __init__(self,
                 size=10000,
                 num_hashes=3):
        self.size       = size
        self.num_hashes = num_hashes
        self.bits       = np.zeros(
            size, dtype=bool)
        self.seeds      = np.random.randint(
            1, 1000, num_hashes)

    def _hashes(self, key):
        return [
            (abs(hash(key)) * int(s))
            % self.size
            for s in self.seeds]

    def add(self, key):
        for pos in self._hashes(key):
            self.bits[pos] = True

    def contains(self, key):
        return all(
            self.bits[pos]
            for pos in self._hashes(key))

    def reset(self):
        self.bits = np.zeros(
            self.size, dtype=bool)


# ─────────────────────────────────────────
# W-TINYLFU QUEUE
# ─────────────────────────────────────────

class WTinyLFUQueue:
    def __init__(self, capacity,
                 window_ratio=WINDOW_CACHE_RATIO):
        self.capacity    = max(1, capacity)
        self.window_size = max(1, int(
            self.capacity * window_ratio))
        self.main_size   = max(1,
            self.capacity - self.window_size)
        self.window_cache      = OrderedDict()
        self.main_cache        = OrderedDict()
        self.count_min_sketch  = CountMinSketch()
        self.bloom_filter      = BloomFilter()
        self.eviction_candidates = []
        self.hits      = 0
        self.misses    = 0
        self.evictions = 0

    def find_available_container(
            self, function_id,
            current_time):
        if function_id in self.window_cache:
            container = (
                self.window_cache[function_id])
            if container.is_available(
                    current_time):
                self.count_min_sketch.add(
                    function_id)
                self.window_cache\
                    .move_to_end(function_id)
                self.hits += 1
                return container

        if function_id in self.main_cache:
            container = (
                self.main_cache[function_id])
            if container.is_available(
                    current_time):
                self.count_min_sketch.add(
                    function_id)
                self.main_cache\
                    .move_to_end(function_id)
                self.hits += 1
                return container

        self.misses += 1
        return None

    def add_container(self, container,
                      current_time):
        function_id = container.function_id
        self.bloom_filter.add(function_id)
        self.count_min_sketch.add(function_id)
        if (len(self.window_cache) >=
                self.window_size):
            self._promote_from_window(
                current_time)
        self.window_cache[function_id] = (
            container)
        self.window_cache.move_to_end(
            function_id)

    def _promote_from_window(self,
                              current_time):
        if not self.window_cache:
            return
        window_victim_id = next(
            iter(self.window_cache))
        window_victim = (
            self.window_cache[
                window_victim_id])
        if not self.bloom_filter.contains(
                window_victim_id):
            del self.window_cache[
                window_victim_id]
            self.evictions += 1
            return
        if (len(self.main_cache) >=
                self.main_size):
            main_victim_id = next(
                iter(self.main_cache))
            window_freq = (
                self.count_min_sketch
                .estimate(window_victim_id))
            main_freq = (
                self.count_min_sketch
                .estimate(main_victim_id))
            if window_freq > main_freq:
                evicted = (
                    self.main_cache.pop(
                        main_victim_id))
                self._handle_eviction(
                    evicted, current_time)
                del self.window_cache[
                    window_victim_id]
                self.main_cache[
                    window_victim_id] = (
                    window_victim)
            else:
                del self.window_cache[
                    window_victim_id]
                self._handle_eviction(
                    window_victim,
                    current_time)
        else:
            del self.window_cache[
                window_victim_id]
            self.main_cache[
                window_victim_id] = (
                window_victim)

    def _handle_eviction(self, container,
                         current_time):
        self.evictions += 1
        if container.state == (
                Container.RUNNING):
            self.eviction_candidates\
                .append(container)

    def resize(self, new_capacity,
               current_time):
        new_capacity     = max(1, new_capacity)
        old_capacity     = self.capacity
        self.capacity    = new_capacity
        self.window_size = max(1, int(
            new_capacity * WINDOW_CACHE_RATIO))
        self.main_size   = max(
            1, new_capacity - self.window_size)
        evicted = []
        if new_capacity < old_capacity:
            while (len(self.window_cache) >
                   self.window_size):
                eid, ec = (
                    self.window_cache
                    .popitem(last=False))
                self._handle_eviction(
                    ec, current_time)
                evicted.append(ec)
            while (len(self.main_cache) >
                   self.main_size):
                eid, ec = (
                    self.main_cache
                    .popitem(last=False))
                self._handle_eviction(
                    ec, current_time)
                evicted.append(ec)
        return evicted

    def get_stats(self):
        total    = self.hits + self.misses
        hit_rate = (self.hits / total * 100
                    if total > 0 else 0)
        return {
            'capacity':    self.capacity,
            'window_size': self.window_size,
            'main_size':   self.main_size,
            'window_used': len(
                self.window_cache),
            'main_used':   len(
                self.main_cache),
            'hits':        self.hits,
            'misses':      self.misses,
            'hit_rate':    hit_rate,
            'evictions':   self.evictions
        }

    def get_total_containers(self):
        return (len(self.window_cache) +
                len(self.main_cache))

    def get_running_containers(self):
        running = 0
        for c in self.window_cache.values():
            if c.state == Container.RUNNING:
                running += 1
        for c in self.main_cache.values():
            if c.state == Container.RUNNING:
                running += 1
        return running

    def get_wasted_memory_time(self,
                                current_time):
        total = 0.0
        for c in self.window_cache.values():
            total += c.get_wasted_memory_time(
                current_time)
        for c in self.main_cache.values():
            total += c.get_wasted_memory_time(
                current_time)
        return total

    def get_all_containers(self):
        """Returns all containers in queue"""
        containers = []
        containers.extend(
            self.window_cache.values())
        containers.extend(
            self.main_cache.values())
        return containers


# ─────────────────────────────────────────
# S-CACHE
# Extended with comprehensive metrics!
# ─────────────────────────────────────────

class SCache:
    """
    Complete S-Cache implementation.
    Extended with professor-recommended
    comprehensive metrics for TASCAR paper!
    """

    def __init__(self,
                 num_queues=NUM_QUEUES,
                 initial_capacities=(
                         INITIAL_QUEUE_CAPACITY)):

        self.num_queues   = num_queues
        self.current_time = 0.0

        self.queues = []
        for k in range(num_queues):
            capacity = initial_capacities[k]
            queue    = WTinyLFUQueue(capacity)
            self.queues.append(queue)

        self.all_containers = []

        # ─────────────────────────────────
        # CORE COUNTERS
        # ─────────────────────────────────
        self.total_requests    = 0
        self.total_cold_starts = 0
        self.total_warm_starts = 0

        # ─────────────────────────────────
        # NEW: LATENCY TRACKING
        # For P95/P99 and ART
        # ─────────────────────────────────
        self.all_response_times   = []
        self.cold_start_latencies = []
        self.warm_response_times  = []

        # ─────────────────────────────────
        # NEW: RESOURCE TRACKING
        # For CUR and RUE
        # ─────────────────────────────────
        self.total_allocated_memory = 0.0
        self.total_used_memory      = 0.0
        self.memory_snapshots       = []

        # ─────────────────────────────────
        # NEW: SLA TRACKING
        # For SVR
        # ─────────────────────────────────
        self.sla_violations  = 0
        self.sla_threshold   = SLA_THRESHOLD

        # ─────────────────────────────────
        # NEW: THROUGHPUT TRACKING
        # For request throughput
        # ─────────────────────────────────
        self.sim_start_time  = None
        self.sim_end_time    = None
        self.completed_reqs  = 0
        self.failed_reqs     = 0

        # ─────────────────────────────────
        # NEW: ENERGY TRACKING
        # For EPR and CO2
        # ─────────────────────────────────
        self.total_energy    = 0.0
        self.power_per_gb    = POWER_PER_GB

        # ─────────────────────────────────
        # NEW: BURST TRACKING
        # For BHE
        # ─────────────────────────────────
        self.burst_window    = []
        self.burst_requests  = 0
        self.burst_served    = 0
        self.burst_threshold = BURST_THRESHOLD

        # ─────────────────────────────────
        # NEW: SCALING TRACKING
        # For SA and Elasticity
        # ─────────────────────────────────
        self.scaling_actions   = []
        self.demanded_capacity = list(
            EXPECTED_DEMAND)
        self.scaling_timestamps = []

    def handle_request(self, function_call):
        """
        Main entry point for every function call.
        Returns True = warm start
        Returns False = cold start
        Extended with comprehensive metrics!
        """
        self.current_time   = (
            function_call.arrival_time)
        self.total_requests += 1

        # Track sim start time
        if self.sim_start_time is None:
            self.sim_start_time = (
                self.current_time)

        self.sim_end_time = self.current_time

        # Burst detection
        self._update_burst_window(
            self.current_time)
        is_burst = (
            len(self.burst_window) >
            self.burst_threshold)
        if is_burst:
            self.burst_requests += 1

        queue_idx = function_call.queue_index
        queue     = self.queues[queue_idx]

        container = (
            queue.find_available_container(
                function_call.function_id,
                self.current_time))

        if container is not None:
            # WARM START
            container.start_execution(
                self.current_time,
                function_call.execution_time)
            self.total_warm_starts += 1
            self.completed_reqs    += 1

            # Response time = execution only
            response_time = (
                function_call.execution_time)
            self.all_response_times.append(
                response_time)
            self.warm_response_times.append(
                response_time)

            # Energy
            mem_gb = (
                function_call.memory_mb /
                1024.0)
            self.total_energy += (
                mem_gb *
                function_call.execution_time *
                self.power_per_gb)

            # Memory tracking
            self.total_used_memory += (
                function_call.memory_mb)
            self.total_allocated_memory += (
                function_call.memory_mb)

            if is_burst:
                self.burst_served += 1

            return True

        else:
            # COLD START
            self.total_cold_starts += 1
            self.completed_reqs    += 1

            # Response time = cold start + execution
            response_time = (
                function_call.cold_start_overhead +
                function_call.execution_time)
            self.all_response_times.append(
                response_time)
            self.cold_start_latencies.append(
                function_call.cold_start_overhead)

            # SLA violation check
            if (function_call.cold_start_overhead >
                    self.sla_threshold):
                self.sla_violations += 1

            # Create new container
            new_container = Container(
                function_id=(
                    function_call.function_id),
                cold_start_overhead=(
                    function_call
                    .cold_start_overhead),
                memory_mb=(
                    function_call.memory_mb),
                current_time=self.current_time)

            queue.add_container(
                new_container,
                self.current_time)
            self.all_containers.append(
                new_container)

            # Energy for cold start + execution
            mem_gb = (
                function_call.memory_mb /
                1024.0)
            self.total_energy += (
                mem_gb *
                response_time *
                self.power_per_gb)

            # Memory tracking
            self.total_used_memory += (
                function_call.memory_mb)
            # Allocated includes overhead
            self.total_allocated_memory += (
                function_call.memory_mb * 1.2)

            return False

    def _update_burst_window(self,
                              current_time):
        """Keep only last second of requests"""
        self.burst_window = [
            t for t in self.burst_window
            if current_time - t < 1.0]
        self.burst_window.append(
            current_time)

    def get_total_wasted_memory_time(self):
        total = 0.0
        for queue in self.queues:
            total += (
                queue.get_wasted_memory_time(
                    self.current_time))
        return total

    def scale_queue(self, queue_idx,
                    scaling_factor):
        queue   = self.queues[queue_idx]
        old_cap = queue.capacity
        new_cap = max(1, int(
            old_cap * (1 + scaling_factor)))
        evicted = queue.resize(
            new_cap, self.current_time)

        # Track scaling action
        self.scaling_actions.append({
            'queue':    queue_idx,
            'old_cap':  old_cap,
            'new_cap':  new_cap,
            'time':     self.current_time
        })

        return evicted

    def get_state(self):
        state = []
        for k, queue in enumerate(
                self.queues):
            stats = queue.get_stats()
            queue_state = [
                stats['capacity'],
                stats['window_used'] +
                stats['main_used'],
                stats['hits'] + stats['misses'],
                stats['misses'],
                stats['evictions'],
                queue.get_running_containers(),
                queue.get_wasted_memory_time(
                    self.current_time)
            ]
            state.extend(queue_state)
        return state

    def get_queue_stats(self):
        return [q.get_stats()
                for q in self.queues]

    def reset_invocation_counters(self):
        for queue in self.queues:
            queue.hits      = 0
            queue.misses    = 0
            queue.evictions = 0

    # ─────────────────────────────────────
    # NEW COMPREHENSIVE METRICS
    # Professor recommended!
    # ─────────────────────────────────────

    def get_cold_start_rate(self):
        """CSR = cold_starts / total × 100"""
        if self.total_requests == 0:
            return 0.0
        return (self.total_cold_starts /
                self.total_requests * 100)

    def get_acsd(self):
        """
        Average Cold Start Delay
        ACSD = sum(cold_latencies) / N
        """
        if not self.cold_start_latencies:
            return 0.0
        return (sum(self.cold_start_latencies) /
                len(self.cold_start_latencies))

    def get_art(self):
        """
        Average Response Time
        ART = sum(all_response_times) / N
        Includes both warm and cold!
        """
        if not self.all_response_times:
            return 0.0
        return (sum(self.all_response_times) /
                len(self.all_response_times))

    def get_p95_latency(self):
        """P95 tail latency"""
        if not self.all_response_times:
            return 0.0
        sorted_times = sorted(
            self.all_response_times)
        idx = int(
            0.95 * len(sorted_times))
        return sorted_times[
            min(idx,
                len(sorted_times) - 1)]

    def get_p99_latency(self):
        """P99 tail latency"""
        if not self.all_response_times:
            return 0.0
        sorted_times = sorted(
            self.all_response_times)
        idx = int(
            0.99 * len(sorted_times))
        return sorted_times[
            min(idx,
                len(sorted_times) - 1)]

    def get_cur(self):
        """
        Container Utilization Rate
        CUR = active_time / total_lifetime
        Higher is better!
        """
        all_containers = []
        for queue in self.queues:
            all_containers.extend(
                queue.get_all_containers())

        if not all_containers:
            return 0.0

        utils = [
            c.get_utilization(
                self.current_time)
            for c in all_containers]
        return (sum(utils) /
                len(utils) * 100)

    def get_rue(self):
        """
        Resource Utilization Efficiency
        RUE = used_memory / allocated_memory
        Higher is better!
        """
        if self.total_allocated_memory <= 0:
            return 0.0
        return min(
            self.total_used_memory /
            self.total_allocated_memory *
            100, 100.0)

    def get_svr(self):
        """
        SLA Violation Rate
        SVR = violations / total × 100
        Lower is better!
        """
        if self.total_requests == 0:
            return 0.0
        return (self.sla_violations /
                self.total_requests * 100)

    def get_throughput(self):
        """
        Request Throughput
        Throughput = completed / time
        Higher is better!
        """
        if (self.sim_start_time is None or
                self.sim_end_time is None):
            return 0.0
        elapsed = (self.sim_end_time -
                   self.sim_start_time)
        if elapsed <= 0:
            return 0.0
        return (self.completed_reqs /
                elapsed)

    def get_ser(self):
        """
        Successful Execution Ratio
        SER = completed / total × 100
        """
        if self.total_requests == 0:
            return 0.0
        return (self.completed_reqs /
                self.total_requests * 100)

    def get_energy_per_request(self):
        """
        Energy per Request in kWh
        EPR = total_energy / total_requests
        Lower is better!
        """
        if self.total_requests == 0:
            return 0.0
        return (self.total_energy /
                self.total_requests)

    def get_co2_estimate(self):
        """
        CO2 Estimate in kg
        CO2 = energy × carbon_intensity
        Lower is better!
        """
        return (self.total_energy *
                CARBON_INTENSITY)

    def get_bhe(self):
        """
        Burst Handling Efficiency
        BHE = burst_served / burst_requests
        Higher is better!
        """
        if self.burst_requests == 0:
            return 100.0
        return (self.burst_served /
                self.burst_requests * 100)

    def get_scaling_accuracy(self):
        """
        Scaling Accuracy
        SA = 1 - |allocated - demanded|
             / demanded
        Higher is better!
        """
        if not self.scaling_actions:
            return 100.0

        accuracies = []
        for action in self.scaling_actions:
            q_idx    = action['queue']
            demanded = self.demanded_capacity[
                q_idx]
            allocated = action['new_cap']
            if demanded > 0:
                sa = 1 - (
                    abs(allocated - demanded)
                    / demanded)
                accuracies.append(
                    max(0, sa) * 100)

        if not accuracies:
            return 100.0
        return (sum(accuracies) /
                len(accuracies))

    def get_elasticity_score(self):
        """
        Elasticity Score
        Measures how responsive scaling is!
        Based on number of scaling actions
        relative to workload changes!
        Higher = more elastic!
        """
        if self.total_requests == 0:
            return 0.0

        # Ratio of scaling actions to
        # every 10000 requests
        actions_per_10k = (
            len(self.scaling_actions) /
            self.total_requests * 10000)

        # Normalize to 0-100
        # Optimal = 1 action per 10k
        # Too few or too many = less elastic
        if actions_per_10k == 0:
            return 0.0
        elif actions_per_10k <= 1:
            return actions_per_10k * 100
        else:
            return max(
                0,
                100 - (actions_per_10k - 1)
                * 10)

    def get_all_metrics(self):
        """
        Returns ALL metrics as dictionary!
        Used by evaluate_tascar.py!
        Complete professor recommended set!
        """
        return {
            # 1. Cold Start Metrics
            'cold_start_rate':
                self.get_cold_start_rate(),
            'avg_cold_start_overhead':
                self.get_acsd(),
            'p95_latency':
                self.get_p95_latency(),
            'p99_latency':
                self.get_p99_latency(),

            # 2. Resource Metrics
            'container_utilization_rate':
                self.get_cur(),
            'resource_utilization_efficiency':
                self.get_rue(),
            'avg_wasted_memory_time':
                self.get_total_wasted_memory_time(),

            # 3. QoS Metrics
            'avg_response_time':
                self.get_art(),
            'p95_latency':
                self.get_p95_latency(),
            'p99_latency':
                self.get_p99_latency(),
            'sla_violation_rate':
                self.get_svr(),

            # 4. Throughput Metrics
            'throughput':
                self.get_throughput(),
            'successful_execution_ratio':
                self.get_ser(),

            # 5. Energy Metrics
            'energy_per_request':
                self.get_energy_per_request(),
            'co2_estimate':
                self.get_co2_estimate(),

            # 6. Scalability Metrics
            'burst_handling_efficiency':
                self.get_bhe(),
            'scaling_accuracy':
                self.get_scaling_accuracy(),
            'elasticity_score':
                self.get_elasticity_score(),
        }


# ─────────────────────────────────────────
# TEST THIS FILE
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Testing scache.py")
    print("Extended with metrics!")
    print("=" * 55)

    from simulator import (
        AzureDataLoader, Simulator)

    scache = SCache()
    print("\nS-Cache created:")
    for k, q in enumerate(scache.queues):
        print(
            f"  Queue {k}: "
            f"capacity={q.capacity}")

    loader = AzureDataLoader()
    calls  = loader.load_day(1)
    test_calls = calls[:5000]
    print(
        f"\nTesting with "
        f"{len(test_calls)} calls...")

    for call in test_calls:
        scache.handle_request(call)

    metrics = scache.get_all_metrics()

    print("\nAll Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nscache.py working correctly!")
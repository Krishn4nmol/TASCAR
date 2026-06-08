# metrics_tracker.py
# Wraps SCache to track comprehensive metrics
# WITHOUT touching core caching logic!
# All professor recommended metrics here!

import time
import numpy as np
from config import (
    SLA_THRESHOLD,
    CARBON_INTENSITY,
    POWER_PER_GB,
    BURST_THRESHOLD,
    TPI_W1, TPI_W2,
    TPI_W3, TPI_W4, TPI_W5,
    EXPECTED_DEMAND,
    NUM_QUEUES)
from scache import SCache


class MetricsTracker:
    """
    Wraps SCache and tracks all
    professor recommended metrics!

    Core SCache is UNCHANGED!
    Only adds metric tracking on top!

    This ensures:
    - Caching behavior identical to CASR
    - All 16+ metrics available
    - No interference with training!
    """

    def __init__(self):
        # Original SCache unchanged!
        self.scache = SCache()

        # ─────────────────────────────
        # SIMULATION TIMING
        # ─────────────────────────────
        self.sim_start_time   = None
        self.sim_end_time     = None
        self.total_requests   = 0
        self.completed_reqs   = 0

        # ─────────────────────────────
        # COLD START METRICS
        # ─────────────────────────────
        self.cold_starts          = 0
        self.warm_hits            = 0
        self.cold_latencies       = []
        self.all_response_times   = []

        # ─────────────────────────────
        # SLA TRACKING
        # ─────────────────────────────
        self.sla_violations  = 0
        self.sla_threshold   = SLA_THRESHOLD

        # ─────────────────────────────
        # RESOURCE TRACKING
        # ─────────────────────────────
        self.total_used_memory      = 0.0
        self.total_allocated_memory = 0.0

        # ─────────────────────────────
        # ENERGY TRACKING
        # ─────────────────────────────
        self.total_energy  = 0.0
        self.power_per_gb  = POWER_PER_GB

        # ─────────────────────────────
        # BURST TRACKING
        # ─────────────────────────────
        self.burst_window    = []
        self.burst_requests  = 0
        self.burst_served    = 0
        self.burst_threshold = BURST_THRESHOLD

        # ─────────────────────────────
        # SCALING TRACKING
        # ─────────────────────────────
        self.scaling_actions   = []
        self.demanded_capacity = list(
            EXPECTED_DEMAND)

    # ─────────────────────────────────
    # DELEGATE TO SCACHE
    # All core methods pass through!
    # ─────────────────────────────────

    def handle_request(self,
                       function_call):
        """
        Handles request via SCache
        AND tracks metrics!
        Core caching logic unchanged!
        """
        self.total_requests += 1

        # Track sim timing
        arrival = function_call.arrival_time
        if self.sim_start_time is None:
            self.sim_start_time = arrival
        self.sim_end_time = arrival

        # Burst detection
        self.burst_window = [
            t for t in self.burst_window
            if arrival - t < 1.0]
        self.burst_window.append(arrival)
        is_burst = (
            len(self.burst_window) >
            self.burst_threshold)
        if is_burst:
            self.burst_requests += 1

        # Call original SCache!
        is_warm = (
            self.scache.handle_request(
                function_call))

        self.completed_reqs += 1

        if is_warm:
            self.warm_hits += 1
            # Warm response = execution only
            resp = (
                function_call.execution_time)
            self.all_response_times.append(
                resp)

            # Energy for warm execution
            mem_gb = (
                function_call.memory_mb /
                1024.0)
            self.total_energy += (
                mem_gb *
                function_call.execution_time
                * self.power_per_gb)

            # Memory tracking
            self.total_used_memory += (
                function_call.memory_mb)
            self.total_allocated_memory += (
                function_call.memory_mb)

            if is_burst:
                self.burst_served += 1

        else:
            self.cold_starts += 1

            # Cold response = overhead + exec
            resp = (
                function_call
                .cold_start_overhead +
                function_call.execution_time)
            self.all_response_times.append(
                resp)
            self.cold_latencies.append(
                function_call
                .cold_start_overhead)

            # SLA check
            if (function_call
                    .cold_start_overhead >
                    self.sla_threshold):
                self.sla_violations += 1

            # Energy for cold + exec
            mem_gb = (
                function_call.memory_mb /
                1024.0)
            self.total_energy += (
                mem_gb * resp *
                self.power_per_gb)

            # Memory tracking
            self.total_used_memory += (
                function_call.memory_mb)
            self.total_allocated_memory += (
                function_call.memory_mb *
                1.2)

        return is_warm

    def get_state(self):
        """Delegates to SCache!"""
        return self.scache.get_state()

    def scale_queue(self,
                    queue_idx,
                    scaling_factor):
        """
        Delegates to SCache
        AND tracks scaling action!
        """
        old_cap = (
            self.scache.queues[
                queue_idx].capacity)

        result = self.scache.scale_queue(
            queue_idx, scaling_factor)

        new_cap = (
            self.scache.queues[
                queue_idx].capacity)

        # Track scaling action
        self.scaling_actions.append({
            'queue':   queue_idx,
            'old_cap': old_cap,
            'new_cap': new_cap,
        })

        return result

    def get_total_wasted_memory_time(
            self):
        """Delegates to SCache!"""
        return (self.scache
                .get_total_wasted_memory_time())

    def get_queue_stats(self):
        """Delegates to SCache!"""
        return self.scache.get_queue_stats()

    def reset_invocation_counters(self):
        """Delegates to SCache!"""
        return (self.scache
                .reset_invocation_counters())

    # ─────────────────────────────────
    # COMPREHENSIVE METRICS
    # All professor recommended!
    # ─────────────────────────────────

    def get_cold_start_rate(self):
        """CSR = cold / total × 100"""
        if self.total_requests == 0:
            return 0.0
        return (self.cold_starts /
                self.total_requests * 100)

    def get_acsd(self):
        """Average Cold Start Delay"""
        if not self.cold_latencies:
            return 0.0
        return (sum(self.cold_latencies) /
                len(self.cold_latencies))

    def get_art(self):
        """Average Response Time"""
        if not self.all_response_times:
            return 0.0
        return (
            sum(self.all_response_times) /
            len(self.all_response_times))

    def get_p95_latency(self):
        """P95 tail latency"""
        if not self.all_response_times:
            return 0.0
        sorted_t = sorted(
            self.all_response_times)
        idx = int(
            0.95 * len(sorted_t))
        return sorted_t[
            min(idx, len(sorted_t)-1)]

    def get_p99_latency(self):
        """P99 tail latency"""
        if not self.all_response_times:
            return 0.0
        sorted_t = sorted(
            self.all_response_times)
        idx = int(
            0.99 * len(sorted_t))
        return sorted_t[
            min(idx, len(sorted_t)-1)]

    def get_cur(self):
        """
        Container Utilization Rate
        Approximated as warm hit ratio!
        CUR = warm_hits / total × 100
        """
        if self.total_requests == 0:
            return 0.0
        return (self.warm_hits /
                self.total_requests * 100)

    def get_rue(self):
        """Resource Utilization Efficiency"""
        if self.total_allocated_memory <= 0:
            return 0.0
        return min(
            self.total_used_memory /
            self.total_allocated_memory *
            100, 100.0)

    def get_svr(self):
        """SLA Violation Rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.sla_violations /
                self.total_requests * 100)

    def get_throughput(self):
        """Requests per second"""
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
        """Successful Execution Ratio"""
        if self.total_requests == 0:
            return 0.0
        return (self.completed_reqs /
                self.total_requests * 100)

    def get_energy_per_request(self):
        """Energy per Request in kWh"""
        if self.total_requests == 0:
            return 0.0
        return (self.total_energy /
                self.total_requests)

    def get_co2_estimate(self):
        """CO2 Estimate in kg"""
        return (self.total_energy *
                CARBON_INTENSITY)

    def get_bhe(self):
        """Burst Handling Efficiency"""
        if self.burst_requests == 0:
            return 100.0
        return (self.burst_served /
                self.burst_requests * 100)

    def get_scaling_accuracy(self):
        """Scaling Accuracy"""
        if not self.scaling_actions:
            return 100.0
        accuracies = []
        for action in self.scaling_actions:
            q_idx     = action['queue']
            demanded  = (
                self.demanded_capacity[
                    q_idx])
            allocated = action['new_cap']
            if demanded > 0:
                sa = 1 - (
                    abs(allocated -
                        demanded) /
                    demanded)
                accuracies.append(
                    max(0, sa) * 100)
        if not accuracies:
            return 100.0
        return (sum(accuracies) /
                len(accuracies))

    def get_elasticity_score(self):
        """Elasticity Score"""
        if self.total_requests == 0:
            return 0.0
        actions_per_10k = (
            len(self.scaling_actions) /
            self.total_requests * 10000)
        if actions_per_10k == 0:
            return 0.0
        elif actions_per_10k <= 1:
            return actions_per_10k * 100
        else:
            return max(
                0,
                100 -
                (actions_per_10k - 1) *
                10)

    def compute_tpi(self,
                    max_throughput=1000.0):
        """TASCAR Performance Index"""
        csr  = min(
            self.get_cold_start_rate() /
            100.0, 1.0)
        wmt  = min(
            self.get_total_wasted_memory_time()
            / 100.0, 1.0)
        tput = min(
            self.get_throughput() /
            max_throughput, 1.0)
        svr  = min(
            self.get_svr() / 100.0, 1.0)
        rue  = min(
            self.get_rue() / 100.0, 1.0)

        tpi = (TPI_W1 * (1 - csr) +
               TPI_W2 * (1 - wmt) +
               TPI_W3 * tput +
               TPI_W4 * (1 - svr) +
               TPI_W5 * rue)
        return round(tpi * 100, 3)

    def get_all_metrics(self):
        """
        Returns ALL metrics as dict!
        Used by evaluate_tascar.py!
        """
        return {
            # Cold Start Metrics
            'cold_start_rate':
                self.get_cold_start_rate(),
            'avg_cold_start_overhead':
                self.get_acsd(),
            'p95_latency':
                self.get_p95_latency(),
            'p99_latency':
                self.get_p99_latency(),

            # Resource Metrics
            'container_utilization_rate':
                self.get_cur(),
            'resource_utilization_efficiency':
                self.get_rue(),
            'avg_wasted_memory_time':
                self.get_total_wasted_memory_time(),

            # QoS Metrics
            'avg_response_time':
                self.get_art(),
            'sla_violation_rate':
                self.get_svr(),

            # Throughput Metrics
            'throughput':
                self.get_throughput(),
            'successful_execution_ratio':
                self.get_ser(),

            # Energy Metrics
            'energy_per_request':
                self.get_energy_per_request(),
            'co2_estimate':
                self.get_co2_estimate(),

            # Scalability Metrics
            'burst_handling_efficiency':
                self.get_bhe(),
            'scaling_accuracy':
                self.get_scaling_accuracy(),
            'elasticity_score':
                self.get_elasticity_score(),
            # Energy total
            'total_energy':
                self.total_energy,
            # Composite
            'tpi': self.compute_tpi(),
        }
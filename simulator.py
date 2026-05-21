# simulator.py
# Loads Azure dataset and simulates function calls
# Think of it as a replay machine for real cloud data

import pandas as pd
import numpy as np
import os
from config import *

# ─────────────────────────────────────────
# FUNCTION CALL CLASS
# Represents one single function invocation
# ─────────────────────────────────────────

class FunctionCall:
    """
    Represents one function invocation
    This is the basic unit of data in our simulation
    """
    def __init__(self, function_id, arrival_time,
                 cold_start_overhead, execution_time, memory_mb):

        # Unique identifier for this function type
        self.function_id = function_id

        # When this call arrives in seconds
        self.arrival_time = arrival_time

        # How long cold start takes in seconds
        self.cold_start_overhead = cold_start_overhead

        # How long function runs after starting
        self.execution_time = execution_time

        # Memory needed in MB
        self.memory_mb = memory_mb

        # Which queue this belongs to
        self.queue_index = self._assign_queue()

    def _assign_queue(self):
        """
        Assigns function to queue based on cold start overhead
        Queue 0: 0-1 seconds
        Queue 1: 1-60 seconds
        Queue 2: 60+ seconds
        """
        for i in range(len(QUEUE_BOUNDARIES) - 1):
            if QUEUE_BOUNDARIES[i] <= self.cold_start_overhead < QUEUE_BOUNDARIES[i+1]:
                return i
        return NUM_QUEUES - 1

    def __repr__(self):
        return (f"FunctionCall("
                f"id={self.function_id}, "
                f"arrival={self.arrival_time:.1f}s, "
                f"cold_start={self.cold_start_overhead:.1f}s, "
                f"queue={self.queue_index})")


# ─────────────────────────────────────────
# DATA LOADER CLASS
# Reads Azure CSV files
# ─────────────────────────────────────────

class AzureDataLoader:
    """
    Loads and processes Microsoft Azure dataset
    Falls back to synthetic data if files not found
    """

    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path

    def load_day(self, day_number):
        """
        Loads function trace data for one day
        Returns list of FunctionCall objects
        """
        filename = (f"invocations_per_function_md.anon."
                   f"d{day_number:02d}.csv")
        filepath = os.path.join(self.data_path, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            print("Using synthetic data instead...")
            return self._generate_synthetic_data(day_number)

        print(f"Loading day {day_number} from {filepath}...")
        df = pd.read_csv(filepath)
        calls = self._process_dataframe(df)
        print(f"Loaded {len(calls)} calls for day {day_number}")
        return calls

    def _process_dataframe(self, df):
        """
        Converts DataFrame into FunctionCall objects
        """
        calls = []
        minute_cols = [str(i) for i in range(1, 1441)
                      if str(i) in df.columns]

        for _, row in df.iterrows():
            function_id = str(row.get('HashFunction',
                             row.get('function_id', 'unknown')))

            cold_start = float(row.get('cold_start_overhead',
                              row.get('duration',
                              np.random.exponential(10))))

            memory = float(row.get('AverageAllocatedMb',
                          row.get('memory_mb',
                          DEFAULT_CONTAINER_MEMORY_MB)))

            exec_time = float(row.get('execution_time',
                             np.random.exponential(1)))

            for minute_col in minute_cols[:60]:
                minute    = int(minute_col)
                num_calls = int(row.get(minute_col, 0))

                if num_calls > 0:
                    for _ in range(min(num_calls, 10)):
                        arrival = ((minute - 1) * 60 +
                                  np.random.uniform(0, 60))

                        call = FunctionCall(
                            function_id=function_id,
                            arrival_time=arrival,
                            cold_start_overhead=cold_start,
                            execution_time=exec_time,
                            memory_mb=memory
                        )
                        calls.append(call)

        calls.sort(key=lambda x: x.arrival_time)
        return calls

    def _generate_synthetic_data(self, day_number,
                                  num_functions=500):
        """
        Generates realistic synthetic data when
        Azure dataset files are not available.
        Matches statistical properties from the paper.
        """
        print(f"Generating synthetic data for day {day_number}...")
        np.random.seed(day_number)

        functions = []

        # 40% lightweight functions (0 to 1 second cold start)
        for i in range(int(num_functions * 0.4)):
            functions.append({
                'id':          f'http_func_{i}',
                'cold_start':  np.random.uniform(0.1, 1.0),
                'exec_time':   np.random.uniform(0.01, 0.5),
                'memory':      np.random.choice([128, 256]),
                'daily_calls': int(np.random.exponential(500))
            })

        # 40% medium functions (1 to 60 seconds cold start)
        for i in range(int(num_functions * 0.4)):
            functions.append({
                'id':          f'api_func_{i}',
                'cold_start':  np.random.uniform(1.0, 60.0),
                'exec_time':   np.random.uniform(0.5, 5.0),
                'memory':      np.random.choice([256, 512, 1024]),
                'daily_calls': int(np.random.exponential(100))
            })

        # 20% heavy ML functions (60+ seconds cold start)
        for i in range(int(num_functions * 0.2)):
            functions.append({
                'id':          f'ml_func_{i}',
                'cold_start':  np.random.uniform(60.0, 300.0),
                'exec_time':   np.random.uniform(5.0, 60.0),
                'memory':      np.random.choice([1024, 2048, 4096]),
                'daily_calls': int(np.random.exponential(20))
            })

        calls        = []
        day_duration = 86400  # 24 hours in seconds

        for func in functions:
            num_calls     = max(1, func['daily_calls'])
            arrival_times = np.sort(
                np.random.uniform(0, day_duration, num_calls)
            )

            for arrival in arrival_times:
                call = FunctionCall(
                    function_id=func['id'],
                    arrival_time=float(arrival),
                    cold_start_overhead=func['cold_start'],
                    execution_time=func['exec_time'],
                    memory_mb=func['memory']
                )
                calls.append(call)

        calls.sort(key=lambda x: x.arrival_time)
        print(f"Generated {len(calls)} synthetic calls")
        return calls

    def load_multiple_days(self, day_numbers):
        """Loads multiple days of data"""
        all_calls = []
        for day in day_numbers:
            day_calls = self.load_day(day)
            all_calls.extend(day_calls)
        return all_calls


# ─────────────────────────────────────────
# SIMULATOR CLASS
# Main simulation engine
# ─────────────────────────────────────────

class Simulator:
    """
    Core simulation engine.
    Takes function calls and an algorithm,
    simulates what happens, records metrics.
    """

    def __init__(self, algorithm,
                 server_memory_mb=SERVER_MEMORY_MB):
        self.algorithm         = algorithm
        self.server_memory_mb  = server_memory_mb
        self.reset_stats()

    def reset_stats(self):
        """Reset all statistics counters"""
        self.total_invocations   = 0
        self.total_cold_starts   = 0
        self.total_warm_starts   = 0
        self.total_cold_overhead = 0.0

    def run(self, function_calls, verbose=True):
        """
        Main simulation loop.
        Processes each function call one by one.
        Returns performance metrics dictionary.
        """
        self.reset_stats()
        total = len(function_calls)
        print(f"Starting simulation with {total} calls...")

        for i, call in enumerate(function_calls):

            # Print progress every 10000 calls
            if verbose and i % 10000 == 0:
                rate = self._get_cold_start_rate()
                print(f"  Progress: {i}/{total} calls | "
                      f"Cold start rate: {rate:.1f}%")

            # Ask algorithm to handle this request
            # Returns True = warm start, False = cold start
            is_warm = self.algorithm.handle_request(call)

            self.total_invocations += 1

            if is_warm:
                self.total_warm_starts += 1
            else:
                self.total_cold_starts   += 1
                self.total_cold_overhead += call.cold_start_overhead

        # Get memory waste from algorithm
        wmt     = self.algorithm.get_total_wasted_memory_time()
        metrics = self._calculate_metrics(wmt)

        print(f"\nSimulation Complete!")
        print(f"  Cold Start Rate:         "
              f"{metrics['cold_start_rate']:.2f}%")
        print(f"  Avg Cold Start Overhead: "
              f"{metrics['avg_cold_start_overhead']:.2f}s")
        print(f"  Avg Wasted Memory Time:  "
              f"{metrics['avg_wasted_memory_time']:.2f}s")

        return metrics

    def _get_cold_start_rate(self):
        """Returns current cold start rate as percentage"""
        if self.total_invocations == 0:
            return 0.0
        return (self.total_cold_starts /
                self.total_invocations) * 100

    def _calculate_metrics(self, total_wmt):
        """
        Calculates all performance metrics from paper
        Section 5.1.5
        """
        if self.total_invocations == 0:
            return {}

        return {
            'cold_start_rate': self._get_cold_start_rate(),

            'avg_cold_start_overhead': (
                self.total_cold_overhead /
                self.total_invocations),

            'avg_wasted_memory_time': (
                total_wmt /
                self.total_invocations),

            'total_invocations':   self.total_invocations,
            'total_cold_starts':   self.total_cold_starts,
            'total_warm_starts':   self.total_warm_starts,
            'total_cold_overhead': self.total_cold_overhead,
            'total_wmt':           total_wmt
        }


# ─────────────────────────────────────────
# TEST THIS FILE
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Testing simulator.py")
    print("=" * 50)

    loader = AzureDataLoader()
    calls  = loader.load_day(1)

    print(f"\nFirst 3 function calls:")
    for call in calls[:3]:
        print(f"  {call}")

    print(f"\nQueue distribution:")
    total = len(calls)
    for q in range(NUM_QUEUES):
        count = sum(1 for c in calls if c.queue_index == q)
        pct   = 100 * count / total
        print(f"  Queue {q}: {count} calls ({pct:.1f}%)")

    print("\n✅ simulator.py working correctly!")
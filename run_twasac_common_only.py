# run_tascar_common_only.py
# Trains TASCAR on Common workload only (Day 1)
# Used for cross-workload generalization experiment
# Tests if TASCAR generalizes to unseen workloads

import config
config.RANDOM_SEED       = 42
config.TASCAR_MODEL_PATH = "trained_model_tascar_common_only/"
config.TASCAR_RESULTS    = "results_tascar_common_only/"
config.TRAIN_DAYS        = [1]  # Common workload only

import runpy
runpy.run_path("train_tascar.py", run_name="__main__")